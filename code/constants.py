import pathlib

# directories
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
RESOURCE_DIR = pathlib.Path(__file__).resolve().parent.parent / "resources"
MODEL_DIR = RESOURCE_DIR / "model"
EMBEDDINGS_DIR = RESOURCE_DIR / "embeddings"
MAPPING_DIR = RESOURCE_DIR / "mapping"

EXTERNAL_RES = (
    pathlib.Path.home() / "External" / "Shared" / "projects" / "nlp" / "resources"
)

EUROSENSE_DIR = EXTERNAL_RES / "EuroSense"
SEW_DIR = pathlib.Path.home() / "External" / "Shared" / "projects" / "nlp" / "resources" / "sew"

# eurosense files
ES_PRECISION = pathlib.Path.joinpath(EUROSENSE_DIR, "es-high-precision.xml")
ES_COVERAGE = pathlib.Path.joinpath(EUROSENSE_DIR, "es-high-coverage.xml")
ES_EN = DATA_DIR / "es-en.xml"
ES_EN_COVERAGE = DATA_DIR / "es-en-coverage.xml"

# sew files
SEW_CONS = SEW_DIR / "sew_conservative"
SEW_CONS_TAR = SEW_DIR / "sew_conservative.tar.gz"

# tom
TOM_EN = DATA_DIR / "evaluation-framework-ims-training.xml"


# data
SENTENCES = DATA_DIR / "sentences.txt"
SENTENCES_CLEAN = DATA_DIR / "sentences_clean.txt"
SENTENCES_COVERAGE = DATA_DIR / "sentences_coverage.txt"
SENTENCES_BIG = DATA_DIR / "sentences_big.txt"
SENTENCES_SEW = SEW_DIR / "sentences_sew.txt"
SENTENCES_SEW_1 = SEW_DIR / "sentences_sew_1.txt"
SENTENCES_SEW_2 = DATA_DIR / "sentences_sew_2.txt"
SENTENCES_SEW_3 = DATA_DIR / "sentences_sew_3.txt"
SENTENCES_SEW_4 = DATA_DIR / "sentences_sew_4.txt"
SENTENCES_SEW_5 = DATA_DIR / "sentences_sew_5.txt"
SENTENCES_TOM = DATA_DIR / "sentences_tom.txt"

# mapping
BN2WN_MAP = MAPPING_DIR / "bn2wn_mapping.txt"
WORDBN_MAPPING = MAPPING_DIR / "word2bn_mapping.txt"
WORDBN_MAPPING_SEW = MAPPING_DIR / "word2bn_mapping_sew.txt"

# model
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "embeddings.vec"
EMBEDDINGS_FILE_CLEAN = EMBEDDINGS_DIR / "embeddings_clean.vec"
MODEL_FILE = MODEL_DIR / "word2vec.model"

# test
COMBINED = RESOURCE_DIR / "combined.tab"
