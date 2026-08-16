# A fork of https://github.com/bbldCVer/SPEED

# Changes:
- Added `gen_frames` argument that allows interpolation with 1, 3, 7, 15 ... `2^n - 1`
  - Recursive interpolation, so quality with degrade with higher `gen_frames`.
