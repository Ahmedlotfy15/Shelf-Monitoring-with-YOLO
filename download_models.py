import gdown
import os

os.makedirs("models", exist_ok=True)

def download_model(url_id, output_path):
    urls = f"https://drive.google.com/uc?id={url_id}"

    gdown.download(urls, output_path)




download_model("1ZxTyAUt2sQCaLX5BM7-Q4xR-aNPkOJcc", "models/product_model.pt")
download_model("1s42_nJnFbzh07kinPhmcYSNMdKfcJEQs", "models/shelves_model.pt")