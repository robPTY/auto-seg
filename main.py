import tifffile
from micro_sam.sam_annotator.annotator_3d import annotator_3d

def main():
    # Some versions expose annotator_3d as a function
    try:
        raw_image = tifffile.imread("50pad_img_xz.tif")
        embedding_path = ""
        model_type = "vit_l_em_organelles"
        if len(embedding_path) > 0:
            annotator_3d(raw_image, embedding_path=embedding_path, model_type=model_type)
        else:
            annotator_3d(raw_image, model_type=model_type)
    except TypeError:
        # fallback for versions where it's a class
        if len(embedding_path) > 0:
            viewer = annotator_3d(raw_image, embedding_path=embedding_path, model_type=model_type)
        else:
            viewer = annotator_3d(raw_image, model_type=model_type)
        return viewer

if __name__ == "__main__":
    main()