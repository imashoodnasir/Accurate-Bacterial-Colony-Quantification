
import numpy as np, cv2
from skimage.morphology import remove_small_objects
from scipy import ndimage as ndi

def refine(prob_map: np.ndarray, th: float = 0.5):
    binm = (prob_map > th).astype(np.uint8)
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    binm = remove_small_objects(binm.astype(bool), 20).astype(np.uint8)
    dist = cv2.distanceTransform(binm, cv2.DIST_L2, 5)
    if dist.max() > 0:
        peaks = (dist > 0.4 * dist.max()).astype(np.uint8)
    else:
        peaks = np.zeros_like(binm)
    markers, _ = ndi.label(peaks)
    if markers.max() == 0:
        return binm
    ws = cv2.watershed(cv2.cvtColor((prob_map*255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                       markers.astype(np.int32))
    seg = (ws > 0).astype(np.uint8)
    return seg
