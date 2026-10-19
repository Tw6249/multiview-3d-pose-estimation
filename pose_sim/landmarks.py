"""Named source landmarks, independent of robot topology and joint limits."""
from .geometry import JOINTS, EDGES

FEET = [f'{side}_{name}' for side in ('left', 'right') for name in ('big_toe', 'small_toe', 'heel')]
PALM = ('wrist', 'thumb_mcp', 'index_mcp', 'middle_mcp', 'pinky_mcp')
HUMAN33 = JOINTS + FEET + [f'{side}_hand_{name}' for side in ('left', 'right') for name in PALM]
# COCO WholeBody: 17 body, 6 feet, 68 face, 21 left hand, 21 right hand.
HUMAN33_INDICES = list(range(23)) + [base+i for base in (91, 112) for i in (0, 2, 5, 9, 17)]
SCHEMAS = {'body17': ('coco17', JOINTS, list(range(17))),
           'wholebody': ('human33', HUMAN33, HUMAN33_INDICES),
           'hybrid': ('human33', HUMAN33, HUMAN33_INDICES)}
LIMB_BONES = [(f'{side}_{a}', f'{side}_{b}') for side in ('left', 'right')
              for a, b in (('shoulder', 'elbow'), ('elbow', 'wrist'), ('hip', 'knee'), ('knee', 'ankle'))]
BONES = LIMB_BONES + [('left_hip', 'right_hip')]


def edges(names):
    result = [(JOINTS[a], JOINTS[b]) for a, b in EDGES]
    for side in ('left', 'right'):
        result += [(f'{side}_{a}', f'{side}_{b}') for a, b in
                   (('ankle', 'heel'), ('heel', 'big_toe'), ('heel', 'small_toe'), ('big_toe', 'small_toe'))]
        result += [(f'{side}_hand_wrist', f'{side}_hand_{finger}_mcp') for finger in ('thumb', 'index', 'middle', 'pinky')]
    return [(names.index(a), names.index(b)) for a, b in result if a in names and b in names]
