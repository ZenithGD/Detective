from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import numpy_image_to_torch, rbd
from lightglue import viz2d
from lightglue import match_pair

import matplotlib.pyplot as plt

from detective.utils.images import read_image

# SuperPoint+LightGlue
extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

# load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
image0 = numpy_image_to_torch(read_image('resources/photos/20231103_132254.jpg')).cuda()
image1 = numpy_image_to_torch(read_image('resources/target.jpg')).cuda()

feats0, feats1, matches01 = match_pair(extractor, matcher, image0, image1)

kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

print(matches.shape)

axes = viz2d.plot_images([image0, image1])
viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)

plt.show()