dirs=("clevr" "cobsat" "matching_mi" "open_mi" "open_t2i_mi" "operator_induction" "operator_induction_interleaved" "textocr")

# Loop through each directory and unzip the files
for dir in "${dirs[@]}"; do
    echo "Unzipping in $dir..."
    unzip "$dir/support.zip" -d "$dir"
    unzip "$dir/query.zip" -d "$dir"
done

echo "Unzipping completed."

