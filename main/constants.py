import os
from pathlib import Path

NUM_2_CHARACTER = {
    0: 'abraham_grampa_simpson',
    1: 'apu_nahasapeemapetilon',
    2: 'bart_simpson',
    3: 'charles_montgomery_burns',
    4: 'chief_wiggum',
    5: 'comic_book_guy',
    6: 'edna_krabappel',
    7: 'homer_simpson',
    8: 'kent_brockman',
    9: 'krusty_the_clown',
    10: 'lisa_simpson',
    11: 'marge_simpson',
    12: 'milhouse_van_houten',
    13: 'moe_szyslak',
    14: 'ned_flanders',
    15: 'nelson_muntz',
    16: 'principal_skinner',
    17: 'sideshow_bob'
}

PROJECT_PATH = str(Path(__file__).parents[1])
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
FIGURE_PATH = os.path.join(PROJECT_PATH, 'figures')
MODEL_PATH = os.path.join(PROJECT_PATH, 'models')
for path in [DATA_PATH, FIGURE_PATH, MODEL_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)



