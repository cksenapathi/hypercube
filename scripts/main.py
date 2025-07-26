import json
from glob import glob
import torchhd


HDV_DIM = 10000


class Codebook:
    def __init__(self, label, num_elem, hdv_dim=HDV_DIM, vsa="BSC"):
        self.label = label
        self.num_elems = num_elem
        self.hdv_dim = hdv_dim
        self.vecs = torchhd.random(self.num_elems, self.hdv_dim, vsa=vsa)

    def get_elem(self, elem_id):
        return self.vecs[elem_id]

#  class HD_CPM -- hyperdim conformally projecting map
# class HD_WAVEDECK -- hyperdim wave-deck; wavelet basis of ST action correlation



# Contain a dual representation
# Raw data and HDV
class Image:
    def __init__(self, img):
        self.img = img
        # Arena allocation is task element
        self.rows = len(img)
        self.cols = len(img[0])

	self.img_codebook = {}

        self.img_hdv_dict = self._pack_hdv_buffer()
    

    def _make_img_codebook(row_range, col_range, color_range):
        # Row and col will be encoded by a single permuted HDV
        self._row_hdv, self._col_hdv = torchhd.random(2, HDV_DIM)
	self._color_hdvs = torchhd.random(10, HDV_DIM)
        return {}
        print("FIX!! main img_codebook")


    # ideally just pointer to img_codebook
    def _pack_hdv_buffer(self, img_codebook):
        self.rows_hdv = img_codebook['rows'][self.rows]
        self.cols_hdv = img_codebook['cols'][self.cols]
        
        return {}

    def _pack_hdv_image(self, img


def main():
    files = glob("data/training/*.json")
    
    for f in files:
        with open(f, 'r') as file:
            data_dict = json.load(file)
            train_data = data_dict["train"]
            test_data = data_dict['test']
            print("Num Train: ", len(train_data))
            print("Num Test: ", len(test_data))


if __name__ == "__main__":
    main()
