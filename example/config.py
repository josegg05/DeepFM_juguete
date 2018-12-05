
# set the path-to-files
#TRAIN_FILE = "./data/train.csv"
#TEST_FILE = "./data/test.csv"

DATA_BASE = "./data/data_final_comp.csv"
DATA_GAME = "./data/games.csv"
TRAIN_FILE = "./data/data_final_train.csv"
TEST_FILE = "./data/data_final_test.csv"

SUB_DIR = "./output"
DATA_DIR = "./data/"
OUTPUT_NAME = "result.csv"


NUM_SPLITS = 3
RANDOM_SEED = 2018

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
    'user_id', 'item_id', 'recommend'

]

NUMERIC_COLS = [
    # # binary
    "CategoryCoop", "CategoryInAppPurchase", "CategoryIncludeLevelEditor",
    "CategoryIncludeSrcSDK", "CategoryMMO", "CategoryMultiplayer",
    "CategorySinglePlayer", "CategoryVRSupport",
    "GenreIsAction", "GenreIsAdventure", "GenreIsCasual",
    "GenreIsEarlyAccess", "GenreIsFreeToPlay", "GenreIsIndie",
    "GenreIsMassivelyMultiplayer", "GenreIsNonGame", "GenreIsRPG",
    "GenreIsRacing", "GenreIsSimulation", "GenreIsSports",
    "GenreIsStrategy",
    "PlatformLinux", "PlatformMac", "PlatformWindows",
    # numeric
    "Metacritic", "RecommendationCount", "count"

]

IGNORE_COLS = [
    "target", "sentiment"
]
