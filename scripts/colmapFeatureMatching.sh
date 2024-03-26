colmap=~/Documents/colmap/build/src/exe/colmap
# Path of the executable program of colmap

# vocab_tree_path=~/Documents/colmap/vocab_tree_flickr100K_words256K.bin
# use vocab_tree_matcher for matching if the dataset is too large to use the exhaustive_matcher
# pretrained vocab_tree can be downloaded here: https://demuc.de/colmap/

runColmapForFeatures(){
    dataset_path=$1
    colmap_path="${dataset_path}/colmap"
    database_path="${colmap_path}/database.db"
    images_path="${dataset_path}/images"
    binary_path="${colmap_path}/0"
    mkdir $colmap_path
    $colmap feature_extractor   --database_path $database_path --image_path $images_path --ImageReader.camera_model PINHOLE
    $colmap exhaustive_matcher --database_path $database_path
    # $colmap vocab_tree_matcher --database_path $database_path  --VocabTreeMatching.vocab_tree_path $vocab_tree_path --SiftMatching.use_gpu 1 --SiftMatching.max_num_matches 31424
    $colmap mapper --database_path $database_path --output_path $colmap_path --image_path $images_path
    $colmap model_converter --input_path $binary_path --output_path $dataset_path --output_type TXT
}

runColmapForFeatures $1

