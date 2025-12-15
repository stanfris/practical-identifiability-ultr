from pathlib import Path
from typing import List, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax.training import checkpoints
from jax import Array
from omegaconf import DictConfig, OmegaConf
from two_tower_confounding.simulation.simulator import Simulator
import os
import pickle as pkl
from hydra.utils import instantiate
from two_tower_confounding.data.base import RatingDataset
from two_tower_confounding.simulation.datasets import ClickDataset

def np_collate(batch):
    """
    Collates a list of dictionaries into a single dict.
    The method assumes that all dicts in the list have the same keys.

    E.g.: batch = [{"query_doc_features": [...]}, {"query_doc_features": [...]}]
    -> {"query_doc_features": [...]}
    """
    keys = batch[0].keys()
    return {key: np.stack([sample[key] for sample in batch]) for key in keys}


def reduce_per_query(loss: Array, where: Array) -> Array:
    loss = loss.reshape(len(loss), -1)
    where = where.reshape(len(where), -1)

    # Adopt Rax safe_reduce as jnp.mean can return NaN if all inputs are 0,
    # which happens easily for pairwise loss functions without any valid pair.
    # Replace NaNs with 0 after reduce, but propagate if the loss already contains NaNs:
    is_input_valid = jnp.logical_not(jnp.any(jnp.isnan(loss)))
    output = jnp.mean(loss, where=where, axis=1)
    output = jnp.where(jnp.isnan(output) & is_input_valid, 0.0, output)

    return output


def collect_metrics(results: List[Dict[str, Array]]) -> pd.DataFrame:
    """
    Collects batches of metrics into a single pandas DataFrame:
    [
        {"ndcg": [0.8, 0.3], "MRR": [0.9, 0.2]},
        {"ndcg": [0.2, 0.1], "MRR": [0.1, 0.02]},
        ...
    ]
    """
    # Convert Jax Arrays to numpy:
    np_results = [dict_to_numpy(r) for r in results]
    # Unroll values in batches into individual rows:
    df = pd.DataFrame(np_results)
    return df.explode(column=list(df.columns)).reset_index(drop=True)


def aggregate_metrics(df: pd.DataFrame, ignore_columns=["query"]) -> Dict[str, float]:
    df = df.drop(columns=ignore_columns, errors="ignore")
    return df.mean(axis=0).to_dict()


def dict_to_numpy(_dict: Dict[str, Array]) -> Dict[str, np.ndarray]:
    return {k: jax.device_get(v) for k, v in _dict.items()}


def save_state(state, directory: Path, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    path = (directory / name).absolute()
    checkpoints.save_checkpoint(path, state, step=0, overwrite=True)


def train_val_test_datasets(config: DictConfig, varying: bool = False):
    """
    Simulate clicks on the original train, val, and test datasets of a LTR dataset.
    Note that this should not be used when using embedding towers, as test query-doc pairs
    do not appear during training. To test embedding towers, use the cross-validation method.
    """

    #### LTR Datasets ####
    dataset = instantiate(config.data.dataset)
    preprocessor = instantiate(config.data.preprocessor)

    train_dataset = preprocessor(dataset.load("train"))
    val_dataset = preprocessor(dataset.load("val"))
    if varying and config.load_test_datasets:
        print("Loading pre-saved test datasets", config.test_dataset_name, config.test_click_dataset_name)
        with open(f"../test_datasets/{config.test_dataset_name}", "rb") as f:
            test_dataset = pkl.load(f)
        with open(f"../test_datasets/{config.test_click_dataset_name}", "rb") as f:
            test_click_dataset = pkl.load(f)
    else:
        test_dataset = preprocessor(dataset.load("test"))

    #### Simulate user clicks ####
    logging_policy_ranker = instantiate(config.logging_policy_ranker)
    logging_policy_ranker.fit(train_dataset)
    logging_policy_sampler = instantiate(config.logging_policy_sampler)

    simulator = Simulator(
        logging_policy_ranker=logging_policy_ranker,
        logging_policy_sampler=logging_policy_sampler,
        bias_strength=config.bias_strength,
        random_state=config.random_state,
    )

    train_click_dataset = simulator(train_dataset, config.train_clicks)
    val_click_dataset = simulator(val_dataset, config.val_clicks)
    if not varying or not config.load_test_datasets:
        test_click_dataset = simulator(test_dataset, config.test_clicks)

    if not varying and config.save_test_datasets:
        print("Saving test datasets", config.test_dataset_name, config.test_click_dataset_name)
        os.makedirs("../test_datasets", exist_ok=True)
        with open(f"../test_datasets/{config.test_dataset_name}", "wb") as f:
            pkl.dump(test_dataset, f)
        with open(f"../test_datasets/{config.test_click_dataset_name}", "wb") as f:
            pkl.dump(test_click_dataset, f)

    return train_click_dataset, val_click_dataset, test_click_dataset, test_dataset



def load_custom_click_dataset(path: str, config: DictConfig) -> ClickDataset:
    dataset_dir = Path(config.dataset_dir).expanduser()
    file_path = dataset_dir / path
    data = np.load(file_path, allow_pickle=True)
    padded_positions = data["padded_positions"]
    mask = data["mask"]
    padded_clicks = data["padded_clicks"]
    sessions_per_query = data["sessions_per_query"]
    sessions_per_doc_pos = data["sessions_per_doc_pos"]
    query_doc_features = data["query_doc_features"]
    lp_query_doc_features = data["lp_query_doc_features"]
    query_doc_ids = data["query_doc_ids"]
    n = data["n"]
    queries = data["queries"]
    unique_list = data["unique_list"]

    rating_dataset = RatingDataset(
        query = queries,
        query_doc_ids=query_doc_ids,
        query_doc_features=query_doc_features,
        lp_query_doc_features=lp_query_doc_features,
        labels=padded_clicks,
        mask=mask,
        n=n,
    )

    remap = True

    # if remap is True, map the 0 index of the lp_query_doc_features to 12 if 12 or larger. 
    if remap:
        lp_query_doc_features[:, :, 0] = np.clip(
        lp_query_doc_features[:, :, 0], None, 12
        )

        media_type_remap = [np.int32(2), np.int32(4), np.int32(5), np.int32(6), np.int32(8), np.int32(9), np.int32(11), np.int32(12), np.int32(13), np.int32(16), np.int32(18), np.int32(19), np.int32(20), np.int32(21), np.int32(22), np.int32(23), np.int32(24), np.int32(25), np.int32(26), np.int32(27), np.int32(28), np.int32(29), np.int32(30), np.int32(31), np.int32(32), np.int32(33), np.int32(34), np.int32(35), np.int32(36), np.int32(37), np.int32(38), np.int32(39), np.int32(40), np.int32(41), np.int32(42), np.int32(43), np.int32(44), np.int32(46), np.int32(47), np.int32(48), np.int32(49), np.int32(50), np.int32(51), np.int32(52), np.int32(53), np.int32(54), np.int32(55), np.int32(56), np.int32(57), np.int32(58), np.int32(59), np.int32(60), np.int32(61), np.int32(62), np.int32(63), np.int32(64), np.int32(65), np.int32(66), np.int32(67), np.int32(68), np.int32(69), np.int32(70), np.int32(71), np.int32(72), np.int32(73), np.int32(74), np.int32(75), np.int32(76), np.int32(77), np.int32(78), np.int32(79), np.int32(80), np.int32(81), np.int32(82), np.int32(83), np.int32(84), np.int32(85), np.int32(86), np.int32(87), np.int32(88), np.int32(89), np.int32(90), np.int32(91), np.int32(92), np.int32(93), np.int32(94), np.int32(95), np.int32(96), np.int32(97), np.int32(98), np.int32(99), np.int32(100), np.int32(101), np.int32(102), np.int32(103), np.int32(104), np.int32(105), np.int32(106), np.int32(107), np.int32(108), np.int32(109), np.int32(110), np.int32(111), np.int32(112), np.int32(113), np.int32(114), np.int32(115), np.int32(116), np.int32(117), np.int32(118), np.int32(119), np.int32(120), np.int32(121), np.int32(122), np.int32(123), np.int32(124), np.int32(125), np.int32(126), np.int32(127), np.int32(128), np.int32(129), np.int32(130), np.int32(131), np.int32(132), np.int32(133), np.int32(134), np.int32(135), np.int32(136), np.int32(137), np.int32(138), np.int32(139), np.int32(140), np.int32(141), np.int32(142), np.int32(143), np.int32(144), np.int32(145), np.int32(146), np.int32(147), np.int32(148), np.int32(149), np.int32(150), np.int32(151), np.int32(152), np.int32(153), np.int32(154), np.int32(155), np.int32(156), np.int32(157), np.int32(158), np.int32(159), np.int32(160), np.int32(161), np.int32(162), np.int32(163), np.int32(164), np.int32(165), np.int32(166), np.int32(167), np.int32(168), np.int32(169), np.int32(170), np.int32(171), np.int32(172), np.int32(173), np.int32(174), np.int32(175), np.int32(176), np.int32(177), np.int32(178), np.int32(179), np.int32(180), np.int32(181), np.int32(182), np.int32(183), np.int32(184), np.int32(185), np.int32(186), np.int32(187), np.int32(188), np.int32(189), np.int32(190), np.int32(191), np.int32(192), np.int32(193), np.int32(194), np.int32(195), np.int32(196), np.int32(197), np.int32(198), np.int32(199), np.int32(200), np.int32(201), np.int32(202), np.int32(203), np.int32(204), np.int32(205), np.int32(206), np.int32(207), np.int32(208), np.int32(209), np.int32(210), np.int32(211), np.int32(212), np.int32(213), np.int32(214), np.int32(215), np.int32(216), np.int32(217), np.int32(218), np.int32(219), np.int32(220), np.int32(221), np.int32(222), np.int32(223), np.int32(224), np.int32(225), np.int32(226), np.int32(227), np.int32(228), np.int32(229), np.int32(230), np.int32(231), np.int32(232), np.int32(233), np.int32(234), np.int32(235), np.int32(236), np.int32(237), np.int32(238), np.int32(239), np.int32(240), np.int32(241), np.int32(242), np.int32(243), np.int32(244), np.int32(247), np.int32(248), np.int32(249), np.int32(250), np.int32(251), np.int32(252), np.int32(253), np.int32(254), np.int32(255), np.int32(256), np.int32(257), np.int32(258), np.int32(259), np.int32(260), np.int32(261), np.int32(262), np.int32(263), np.int32(264), np.int32(265), np.int32(266), np.int32(267), np.int32(268), np.int32(269), np.int32(270), np.int32(271), np.int32(272), np.int32(273), np.int32(274), np.int32(275), np.int32(276), np.int32(277), np.int32(278), np.int32(279), np.int32(280), np.int32(281), np.int32(282), np.int32(283), np.int32(284), np.int32(285), np.int32(286), np.int32(287), np.int32(288), np.int32(289), np.int32(290), np.int32(291), np.int32(292), np.int32(293), np.int32(294), np.int32(295), np.int32(296), np.int32(297), np.int32(298), np.int32(299), np.int32(300), np.int32(301), np.int32(302), np.int32(303), np.int32(304), np.int32(305), np.int32(306), np.int32(307), np.int32(308), np.int32(309), np.int32(310), np.int32(311), np.int32(312), np.int32(313), np.int32(314), np.int32(315), np.int32(316), np.int32(317), np.int32(318), np.int32(319), np.int32(320), np.int32(321), np.int32(322), np.int32(323), np.int32(324), np.int32(325), np.int32(326), np.int32(327), np.int32(328), np.int32(329), np.int32(330), np.int32(331), np.int32(332), np.int32(333), np.int32(334), np.int32(335), np.int32(336), np.int32(337), np.int32(338), np.int32(339), np.int32(340), np.int32(341), np.int32(342), np.int32(343), np.int32(344), np.int32(345), np.int32(346), np.int32(347), np.int32(348), np.int32(349), np.int32(350), np.int32(351), np.int32(352), np.int32(353), np.int32(354), np.int32(355), np.int32(356), np.int32(357), np.int32(358), np.int32(359), np.int32(360), np.int32(361), np.int32(362), np.int32(363), np.int32(364), np.int32(365), np.int32(366), np.int32(367), np.int32(368), np.int32(369), np.int32(370), np.int32(371), np.int32(372), np.int32(373), np.int32(374), np.int32(375), np.int32(376), np.int32(377), np.int32(378), np.int32(379), np.int32(380), np.int32(381), np.int32(382), np.int32(383), np.int32(384), np.int32(385), np.int32(386), np.int32(387), np.int32(388), np.int32(389), np.int32(390), np.int32(391), np.int32(392), np.int32(393), np.int32(394), np.int32(395), np.int32(396), np.int32(397), np.int32(398), np.int32(399), np.int32(400), np.int32(401), np.int32(402), np.int32(403), np.int32(404), np.int32(405), np.int32(406), np.int32(407), np.int32(408), np.int32(409), np.int32(410), np.int32(411), np.int32(412), np.int32(413), np.int32(414), np.int32(415), np.int32(416), np.int32(417), np.int32(418), np.int32(419), np.int32(420), np.int32(421), np.int32(422), np.int32(423), np.int32(424), np.int32(425), np.int32(426), np.int32(427), np.int32(428), np.int32(429), np.int32(430), np.int32(431), np.int32(432), np.int32(433), np.int32(434), np.int32(435), np.int32(436), np.int32(437), np.int32(438), np.int32(439), np.int32(440), np.int32(441), np.int32(442), np.int32(443), np.int32(444), np.int32(445), np.int32(446), np.int32(447), np.int32(448), np.int32(449), np.int32(450), np.int32(451), np.int32(452), np.int32(453), np.int32(454), np.int32(455), np.int32(456), np.int32(457), np.int32(458), np.int32(459), np.int32(460), np.int32(461), np.int32(462), np.int32(463), np.int32(464), np.int32(465), np.int32(466), np.int32(467), np.int32(468), np.int32(469), np.int32(470), np.int32(471), np.int32(472), np.int32(473), np.int32(474), np.int32(475), np.int32(476), np.int32(477), np.int32(478), np.int32(479), np.int32(480), np.int32(481), np.int32(482), np.int32(483), np.int32(484), np.int32(485), np.int32(486), np.int32(487), np.int32(488), np.int32(489), np.int32(490), np.int32(491), np.int32(492), np.int32(493), np.int32(494), np.int32(495), np.int32(496), np.int32(497), np.int32(498), np.int32(499), np.int32(500), np.int32(501), np.int32(502), np.int32(503), np.int32(504), np.int32(505), np.int32(506), np.int32(508), np.int32(509), np.int32(510), np.int32(511), np.int32(512), np.int32(513), np.int32(514), np.int32(515), np.int32(516), np.int32(517), np.int32(518), np.int32(519), np.int32(520), np.int32(521), np.int32(522), np.int32(523), np.int32(524), np.int32(525), np.int32(526), np.int32(527), np.int32(528), np.int32(529), np.int32(530), np.int32(531), np.int32(532), np.int32(533), np.int32(534), np.int32(535), np.int32(536), np.int32(537), np.int32(538), np.int32(539), np.int32(540), np.int32(541), np.int32(542), np.int32(543), np.int32(544), np.int32(545), np.int32(546), np.int32(547), np.int32(548), np.int32(549), np.int32(550), np.int32(551), np.int32(552), np.int32(553), np.int32(554), np.int32(555), np.int32(556), np.int32(557), np.int32(558), np.int32(559), np.int32(560), np.int32(561), np.int32(562), np.int32(563), np.int32(564), np.int32(565), np.int32(566), np.int32(567), np.int32(568), np.int32(569), np.int32(570), np.int32(571), np.int32(572), np.int32(573), np.int32(574), np.int32(575), np.int32(576), np.int32(577), np.int32(578), np.int32(579), np.int32(580), np.int32(581), np.int32(582), np.int32(583), np.int32(584), np.int32(585), np.int32(586), np.int32(587), np.int32(588), np.int32(589), np.int32(590), np.int32(591), np.int32(592), np.int32(593), np.int32(594), np.int32(595), np.int32(596), np.int32(597), np.int32(598), np.int32(599), np.int32(600), np.int32(601), np.int32(602), np.int32(603), np.int32(604), np.int32(605), np.int32(606), np.int32(607), np.int32(608), np.int32(609), np.int32(610), np.int32(611), np.int32(612)]

        mask = np.isin(lp_query_doc_features[:, :, 1], media_type_remap)
        lp_query_doc_features[:, :, 1][mask] = 2
    # -----------------------------
    # Construct ClickDataset
    # -----------------------------
    sessions = np.arange(len(rating_dataset))  # each session corresponds to a row in RatingDataset

    click_dataset = ClickDataset(
        rating_dataset=rating_dataset,
        sessions=sessions,
        clicks=padded_clicks,
        positions=padded_positions,
        sessions_per_query=sessions_per_query,
        sessions_per_doc_pos=sessions_per_doc_pos,
    )

    print("RatingDataset.query.shape:", rating_dataset.query.shape)
    print("RatingDataset.query_doc_features.shape:", rating_dataset.query_doc_features.shape)
    print("RatingDataset.lp_query_doc_features.shape:", rating_dataset.lp_query_doc_features.shape)
    print("ClickDataset.clicks.shape:", click_dataset.clicks.shape)
    print("ClickDataset.positions.shape:", click_dataset.positions.shape)
    print("ClickDataset.sessions_per_query.shape:", click_dataset.sessions_per_query.shape)
    print("ClickDataset.sessions_per_doc_pos.shape:", click_dataset.sessions_per_doc_pos.shape)
    return rating_dataset, click_dataset, unique_list
