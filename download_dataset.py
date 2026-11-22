from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openvla/modified_libero_rlds",
    repo_type="dataset",
    local_dir="./data"
)