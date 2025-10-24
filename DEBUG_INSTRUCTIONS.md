# Debug Instructions for Tensor Dimension Error

## What I've Done
I've added comprehensive debug logging to the `stitch_single_optimized` method in `inpaint_cropandstitch.py` to trace exactly where the tensor dimension mismatch is occurring.

## How to Use
1. Run your workflow again with the same settings
2. Check the terminal/console output for the DEBUG messages
3. The debug output will show:
   - Initial tensor shapes
   - Resize operations and their results
   - Mask dimension transformations
   - Canvas extraction coordinates and validation
   - The exact shapes of tensors before the blending operation
   - Which specific operation fails

## What to Look For
The debug output will help identify:
- Whether the canvas_crop extraction is getting the wrong dimensions
- If the mask reshaping is working correctly
- If there's an issue with the coordinates being passed
- The exact tensor shapes at the point of failure

## Expected Debug Output Structure
```
================================================================================
DEBUG: stitch_single_optimized - START
  canvas_image shape: torch.Size([...])
  inpainted_image shape: torch.Size([...])
  mask shape: torch.Size([...])
  ctc (x,y,w,h): (...)
  cto (x,y,w,h): (...)
================================================================================

DEBUG: Resize operations:
  Input image dims: h=X, w=Y
  Target dims: h=X, w=Y
  [resize details...]

DEBUG: Mask shape adjustments:
  [mask transformation details...]

DEBUG: Before canvas_crop extraction:
  canvas_image shape: torch.Size([...])
  Extracting region: y=X:Y, x=X:Y
  canvas_crop shape after extraction: torch.Size([...])
  Expected shape: (1, H, W, 3)

DEBUG: Before blending operation:
  resized_mask shape: torch.Size([...])
  resized_image shape: torch.Size([...])
  canvas_crop shape: torch.Size([...])

DEBUG: Attempting blend operation...
  [will show which operation fails]
```

## Share the Output
Please copy the full DEBUG output from the terminal and share it so we can see exactly what's happening with the tensor dimensions.
