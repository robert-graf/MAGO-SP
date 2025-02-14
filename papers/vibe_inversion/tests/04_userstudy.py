import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[-3]))
sys.path.append(str(Path(__file__).parents[-4]))
sys.path.append(str(Path(__file__).parents[-5]))
sys.path.append(str(Path(__file__).parents[-6]))
sys.path.append(str(Path(__file__).parents[-7]))


from datasets.mevibe import MEVIBE_dataset
from datasets.vibe import VIBE_dataset

MagneticFieldStrength = 3

c_mevibe = MEVIBE_dataset(256, gray=True, test=True, validation=False, create_dataset=False)
c_vibe = VIBE_dataset(256, gray=True, test=True, validation=False, create_dataset=True)
gpu = 0
ddevice = "cuda"
override = False
steps_signal_prior = 50
derivative = "derivatives_inversion"
non_strict_mode = False


random.seed(42)
random.shuffle(c_mevibe.subjects)
print(c_mevibe.subjects[:102])


random.seed(42)
random.shuffle(c_vibe.subjects)
print(c_vibe.subjects[:100])

exit()
