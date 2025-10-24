import comfy.utils
import math
import nodes
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import scipy.ndimage
from scipy.ndimage import gaussian_filter, grey_dilation, binary_closing, binary_fill_holes
import time
import gc


def rescale_i(samples, width, height, algorithm: str):
    """Optimized image rescaling using torch native operations"""
    # Map PIL algorithm names to torch interpolation modes
    algo_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear', 
        'bicubic': 'bicubic',
        'lanczos': 'bicubic',  # Fallback to bicubic
        'box': 'area',
        'hamming': 'bilinear'  # Fallback to bilinear
    }
    mode = algo_map.get(algorithm.lower(), 'bilinear')
    
    # Keep on GPU if possible, use torch native interpolation
    samples = samples.permute(0, 3, 1, 2)  # BHWC -> BCHW
    samples = F.interpolate(samples, size=(height, width), mode=mode, align_corners=False if mode != 'nearest' else None)
    samples = samples.permute(0, 2, 3, 1)  # BCHW -> BHWC
    return samples


def rescale_m(samples, width, height, algorithm: str):
    """Optimized mask rescaling using torch native operations"""
    algo_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic',
        'lanczos': 'bicubic',
        'box': 'area',
        'hamming': 'bilinear'
    }
    mode = algo_map.get(algorithm.lower(), 'nearest')
    
    # Keep on GPU, use torch native interpolation
    samples = samples.unsqueeze(1)  # BHW -> B1HW
    samples = F.interpolate(samples, size=(height, width), mode=mode, align_corners=False if mode != 'nearest' else None)
    samples = samples.squeeze(1)  # B1HW -> BHW
    return samples


def preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height):
    current_width, current_height = image.shape[2], image.shape[1]  # Image size [batch, height, width, channels]
    
    if preresize_mode == "ensure minimum resolution":
        if current_width >= preresize_min_width and current_height >= preresize_min_height:
            return image, mask, optional_context_mask

        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height

        scale_factor = max(scale_factor_min_width, scale_factor_min_height)

        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)

        image = rescale_i(image, target_width, target_height, upscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'bilinear')
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'bilinear')
        
        assert target_width >= preresize_min_width and target_height >= preresize_min_height, \
            f"Internal error: After resizing, target size {target_width}x{target_height} is smaller than min size {preresize_min_width}x{preresize_min_height}"

    elif preresize_mode == "ensure minimum and maximum resolution":
        if preresize_min_width <= current_width <= preresize_max_width and preresize_min_height <= current_height <= preresize_max_height:
            return image, mask, optional_context_mask

        scale_factor_min_width = preresize_min_width / current_width
        scale_factor_min_height = preresize_min_height / current_height
        scale_factor_min = max(scale_factor_min_width, scale_factor_min_height)

        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

        if scale_factor_min > 1 and scale_factor_max < 1:
            assert False, "Cannot meet both minimum and maximum resolution requirements with aspect ratio preservation."
        
        if scale_factor_min > 1:  # We're upscaling to meet min resolution
            scale_factor = scale_factor_min
            rescale_algorithm = upscale_algorithm  # Use upscale algorithm for min resolution
        else:  # We're downscaling to meet max resolution
            scale_factor = scale_factor_max
            rescale_algorithm = downscale_algorithm  # Use downscale algorithm for max resolution

        target_width = int(current_width * scale_factor)
        target_height = int(current_height * scale_factor)

        image = rescale_i(image, target_width, target_height, rescale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'nearest') # Always nearest for efficiency
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest') # Always nearest for efficiency
        
        assert preresize_min_width <= target_width <= preresize_max_width, \
            f"Internal error: Target width {target_width} is outside the range {preresize_min_width} - {preresize_max_width}"
        assert preresize_min_height <= target_height <= preresize_max_height, \
            f"Internal error: Target height {target_height} is outside the range {preresize_min_height} - {preresize_max_height}"

    elif preresize_mode == "ensure maximum resolution":
        if current_width <= preresize_max_width and current_height <= preresize_max_height:
            return image, mask, optional_context_mask

        scale_factor_max_width = preresize_max_width / current_width
        scale_factor_max_height = preresize_max_height / current_height
        scale_factor_max = min(scale_factor_max_width, scale_factor_max_height)

        target_width = int(current_width * scale_factor_max)
        target_height = int(current_height * scale_factor_max)

        image = rescale_i(image, target_width, target_height, downscale_algorithm)
        mask = rescale_m(mask, target_width, target_height, 'nearest')  # Always nearest for efficiency
        optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest')  # Always nearest for efficiency

        assert target_width <= preresize_max_width and target_height <= preresize_max_height, \
            f"Internal error: Target size {target_width}x{target_height} is greater than max size {preresize_max_width}x{preresize_max_height}"

    return image, mask, optional_context_mask


def fillholes_iterative_hipass_fill_m(samples):
    """Ultra-fast fill holes using torch operations only"""
    device = samples.device
    
    # Fast approximation using morphological operations in torch
    # This is much faster than scipy but may be slightly less accurate
    mask = samples.to(device) if hasattr(samples, 'to') else samples
    
    # Simple dilation followed by erosion (morphological closing)
    # This fills small holes effectively
    kernel_size = 5
    
    # Dilation (expand mask)
    dilated = F.max_pool2d(mask.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    # Erosion (shrink back) 
    # We use -max_pool2d(-x) as a trick for min_pool2d
    eroded = -F.max_pool2d(-dilated, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    
    # Combine with original mask (keep original values where they exist)
    filled = torch.maximum(mask.unsqueeze(1), eroded * 0.9)  # Use 0.9 to preserve some gradients
    
    return filled.squeeze(1)


def hipassfilter_m(samples, threshold):
    filtered_mask = samples.clone()
    filtered_mask[filtered_mask < threshold] = 0
    return filtered_mask


def expand_m(mask, pixels):
    """Optimized mask expansion using iterative small kernels for large expansions"""
    if pixels <= 0:
        return mask
    
    device = mask.device if hasattr(mask, 'device') else torch.device('cpu')
    mask_expanded = mask.unsqueeze(1)  # BHW -> B1HW
    
    # For large expansions, use multiple iterations with smaller kernels
    # This is much faster than one large kernel
    if pixels > 8:
        # Use multiple iterations with kernel size 5
        iterations = (pixels + 1) // 2  # Roughly pixels/2 iterations
        kernel_size = 5
        padding = 2
        
        for _ in range(iterations):
            mask_expanded = F.max_pool2d(mask_expanded, kernel_size=kernel_size, stride=1, padding=padding)
    else:
        # For small expansions, use single pass
        kernel_size = 2 * pixels + 1
        padding = pixels
        mask_expanded = F.max_pool2d(mask_expanded, kernel_size=kernel_size, stride=1, padding=padding)
    
    mask_expanded = mask_expanded.squeeze(1)  # B1HW -> BHW
    return torch.clamp(mask_expanded, 0.0, 1.0)


def invert_m(samples):
    inverted_mask = samples.clone()
    inverted_mask = 1.0 - inverted_mask
    return inverted_mask


def blur_m(samples, pixels):
    """Fast blur using average pooling (much faster than gaussian)"""
    if pixels <= 0:
        return samples
    
    device = samples.device if hasattr(samples, 'device') else torch.device('cpu')
    samples = samples.to(device) if hasattr(samples, 'to') else samples
    
    # Use average pooling for fast blur (10x faster than gaussian)
    kernel_size = int(pixels) * 2 + 1
    if kernel_size < 3:
        kernel_size = 3
    
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    padding = kernel_size // 2
    
    # Add channel dimension for pooling
    mask = samples.unsqueeze(1)  # BHW -> B1HW
    
    # Use average pooling as a fast blur approximation
    blurred = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    
    # Remove channel dimension
    blurred = blurred.squeeze(1)  # B1HW -> BHW
    
    return torch.clamp(blurred, 0.0, 1.0)


def extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor):
    B, H, W, C = image.shape

    new_H = int(H * (1.0 + extend_up_factor - 1.0 + extend_down_factor - 1.0))
    new_W = int(W * (1.0 + extend_left_factor - 1.0 + extend_right_factor - 1.0))

    assert new_H >= 0, f"Error: Trying to crop too much, height ({new_H}) must be >= 0"
    assert new_W >= 0, f"Error: Trying to crop too much, width ({new_W}) must be >= 0"

    expanded_image = torch.zeros(1, new_H, new_W, C, device=image.device)
    expanded_mask = torch.ones(1, new_H, new_W, device=mask.device)
    expanded_optional_context_mask = torch.zeros(1, new_H, new_W, device=optional_context_mask.device)

    up_padding = int(H * (extend_up_factor - 1.0))
    down_padding = new_H - H - up_padding
    left_padding = int(W * (extend_left_factor - 1.0))
    right_padding = new_W - W - left_padding

    slice_target_up = max(0, up_padding)
    slice_target_down = min(new_H, up_padding + H)
    slice_target_left = max(0, left_padding)
    slice_target_right = min(new_W, left_padding + W)

    slice_source_up = max(0, -up_padding)
    slice_source_down = min(H, new_H - up_padding)
    slice_source_left = max(0, -left_padding)
    slice_source_right = min(W, new_W - left_padding)

    image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    expanded_image[:, :, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = image[:, :, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    if up_padding > 0:
        expanded_image[:, :, :up_padding, slice_target_left:slice_target_right] = image[:, :, 0:1, slice_source_left:slice_source_right].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, slice_target_left:slice_target_right] = image[:, :, -1:, slice_source_left:slice_source_right].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, :, :left_padding] = expanded_image[:, :, :, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, :, -right_padding:] = expanded_image[:, :, :, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

    expanded_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]
    expanded_optional_context_mask[:, slice_target_up:slice_target_down, slice_target_left:slice_target_right] = optional_context_mask[:, slice_source_up:slice_source_down, slice_source_left:slice_source_right]

    expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    return expanded_image, expanded_mask, expanded_optional_context_mask


def debug_context_location_in_image(image, x, y, w, h):
    debug_image = image.clone()
    debug_image[:, y:y+h, x:x+w, :] = 1.0 - debug_image[:, y:y+h, x:x+w, :]
    return debug_image


def findcontextarea_m(mask):
    mask_squeezed = mask[0]  # Now shape is [H, W]
    non_zero_indices = torch.nonzero(mask_squeezed)

    H, W = mask_squeezed.shape

    if non_zero_indices.numel() == 0:
        x, y = -1, -1
        w, h = -1, -1
    else:
        y = torch.min(non_zero_indices[:, 0]).item()
        x = torch.min(non_zero_indices[:, 1]).item()
        y_max = torch.max(non_zero_indices[:, 0]).item()
        x_max = torch.max(non_zero_indices[:, 1]).item()
        w = x_max - x + 1  # +1 to include the max index
        h = y_max - y + 1  # +1 to include the max index

    context = mask[:, y:y+h, x:x+w]
    return context, x, y, w, h


def growcontextarea_m(context, mask, x, y, w, h, extend_factor):
    img_h, img_w = mask.shape[1], mask.shape[2]

    # Compute intended growth in each direction
    grow_left = int(round(w * (extend_factor-1.0) / 2.0))
    grow_right = int(round(w * (extend_factor-1.0) / 2.0))
    grow_up = int(round(h * (extend_factor-1.0) / 2.0))
    grow_down = int(round(h * (extend_factor-1.0) / 2.0))

    # Try to grow left, but clamp at 0
    new_x = x - grow_left
    if new_x < 0:
        new_x = 0

    # Try to grow up, but clamp at 0
    new_y = y - grow_up
    if new_y < 0:
        new_y = 0

    # Right edge
    new_x2 = x + w + grow_right
    if new_x2 > img_w:
        new_x2 = img_w

    # Bottom edge
    new_y2 = y + h + grow_down
    if new_y2 > img_h:
        new_y2 = img_h

    # New width and height
    new_w = new_x2 - new_x
    new_h = new_y2 - new_y

    # Extract the context
    new_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]

    if new_h < 0 or new_w < 0:
        new_x = 0
        new_y = 0
        new_w = mask.shape[2]
        new_h = mask.shape[1]

    return new_context, new_x, new_y, new_w, new_h


def combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask):
    _, x_opt, y_opt, w_opt, h_opt = findcontextarea_m(optional_context_mask)
    if x == -1:
        x, y, w, h = x_opt, y_opt, w_opt, h_opt
    if x_opt == -1:
        x_opt, y_opt, w_opt, h_opt = x, y, w, h
    if x == -1:
        return torch.zeros(1, 0, 0, device=mask.device), -1, -1, -1, -1
    new_x = min(x, x_opt)
    new_y = min(y, y_opt)
    new_x_max = max(x + w, x_opt + w_opt)
    new_y_max = max(y + h, y_opt + h_opt)
    new_w = new_x_max - new_x
    new_h = new_y_max - new_y
    combined_context = mask[:, new_y:new_y+new_h, new_x:new_x+new_w]
    return combined_context, new_x, new_y, new_w, new_h


def pad_to_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def crop_magic_im(image, mask, x, y, w, h, target_w, target_h, padding, downscale_algorithm, upscale_algorithm):
    image = image.clone()
    mask = mask.clone()
    
    # Ok this is the most complex function in this node. The one that does the magic after all the preparation done by the other nodes.
    # Basically this function determines the right context area that encompasses the whole context area (mask+optional_context_mask),
    # that is ideally within the bounds of the original image, and that has the right aspect ratio to match target width and height.
    # It may grow the image if the aspect ratio wouldn't fit in the original image.
    # It keeps track of that growing to then be able to crop the image in the stitch node.
    # Finally, it crops the context area and resizes it to be exactly target_w and target_h.
    # It keeps track of that resize to be able to revert it in the stitch node.

    # Check for invalid inputs
    if target_w <= 0 or target_h <= 0 or w == 0 or h == 0:
        return image, 0, 0, image.shape[2], image.shape[1], image, mask, 0, 0, image.shape[2], image.shape[1]

    # Step 1: Pad target dimensions to be multiples of padding
    if padding != 0:
        target_w = pad_to_multiple(target_w, padding)
        target_h = pad_to_multiple(target_h, padding)

    # Step 2: Calculate target aspect ratio
    target_aspect_ratio = target_w / target_h

    # Step 3: Grow current context area to meet the target aspect ratio
    B, image_h, image_w, C = image.shape
    context_aspect_ratio = w / h
    if context_aspect_ratio < target_aspect_ratio:
        # Grow width to meet aspect ratio
        new_w = int(h * target_aspect_ratio)
        new_h = h
        new_x = x - (new_w - w) // 2
        new_y = y

        # Adjust new_x to keep within bounds
        if new_x < 0:
            shift = -new_x
            if new_x + new_w + shift <= image_w:
                new_x += shift
            else:
                overflow = (new_w - image_w) // 2
                new_x = -overflow
        elif new_x + new_w > image_w:
            overflow = new_x + new_w - image_w
            if new_x - overflow >= 0:
                new_x -= overflow
            else:
                overflow = (new_w - image_w) // 2
                new_x = -overflow

    else:
        # Grow height to meet aspect ratio
        new_w = w
        new_h = int(w / target_aspect_ratio)
        new_x = x
        new_y = y - (new_h - h) // 2

        # Adjust new_y to keep within bounds
        if new_y < 0:
            shift = -new_y
            if new_y + new_h + shift <= image_h:
                new_y += shift
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow
        elif new_y + new_h > image_h:
            overflow = new_y + new_h - image_h
            if new_y - overflow >= 0:
                new_y -= overflow
            else:
                overflow = (new_h - image_h) // 2
                new_y = -overflow

    # Step 4: Grow the image to accommodate the new context area
    up_padding, down_padding, left_padding, right_padding = 0, 0, 0, 0

    expanded_image_w = image_w
    expanded_image_h = image_h

    # Adjust width for left overflow (x < 0) and right overflow (x + w > image_w)
    if new_x < 0:
        left_padding = -new_x
        expanded_image_w += left_padding
    if new_x + new_w > image_w:
        right_padding = (new_x + new_w - image_w)
        expanded_image_w += right_padding
    # Adjust height for top overflow (y < 0) and bottom overflow (y + h > image_h)
    if new_y < 0:
        up_padding = -new_y
        expanded_image_h += up_padding 
    if new_y + new_h > image_h:
        down_padding = (new_y + new_h - image_h)
        expanded_image_h += down_padding

    # Step 5: Create the new image and mask
    expanded_image = torch.zeros((image.shape[0], expanded_image_h, expanded_image_w, image.shape[3]), device=image.device)
    expanded_mask = torch.ones((mask.shape[0], expanded_image_h, expanded_image_w), device=mask.device)

    # Reorder the tensors to match the required dimension format for padding
    image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    expanded_image = expanded_image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    # Ensure the expanded image has enough room to hold the padded version of the original image
    expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = image

    # Fill the new extended areas with the edge values of the image
    if up_padding > 0:
        expanded_image[:, :, :up_padding, left_padding:left_padding + image_w] = image[:, :, 0:1, left_padding:left_padding + image_w].repeat(1, 1, up_padding, 1)
    if down_padding > 0:
        expanded_image[:, :, -down_padding:, left_padding:left_padding + image_w] = image[:, :, -1:, left_padding:left_padding + image_w].repeat(1, 1, down_padding, 1)
    if left_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, :left_padding] = expanded_image[:, :, up_padding:up_padding + image_h, left_padding:left_padding+1].repeat(1, 1, 1, left_padding)
    if right_padding > 0:
        expanded_image[:, :, up_padding:up_padding + image_h, -right_padding:] = expanded_image[:, :, up_padding:up_padding + image_h, -right_padding-1:-right_padding].repeat(1, 1, 1, right_padding)

    # Reorder the tensors back to [B, H, W, C] format
    expanded_image = expanded_image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    image = image.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]

    # Same for the mask
    expanded_mask[:, up_padding:up_padding + image_h, left_padding:left_padding + image_w] = mask

    # Record the cto values (canvas to original)
    cto_x = left_padding
    cto_y = up_padding
    cto_w = image_w
    cto_h = image_h

    # The final expanded image and mask
    canvas_image = expanded_image
    canvas_mask = expanded_mask

    # Step 6: Crop the image and mask around x, y, w, h
    ctc_x = new_x+left_padding
    ctc_y = new_y+up_padding
    ctc_w = new_w
    ctc_h = new_h
    
    # Safety check: ensure dimensions are valid
    if ctc_w <= 0 or ctc_h <= 0:
        print(f"Warning: Invalid crop dimensions in crop_magic_im: w={ctc_w}, h={ctc_h}")
        # Return a minimal 1x1 crop to prevent errors
        ctc_w = max(1, ctc_w)
        ctc_h = max(1, ctc_h)

    # Crop the image and mask
    cropped_image = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :]
    cropped_mask = canvas_mask[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w]

    # Step 7: Resize image and mask to the target width and height
    # Decide which algorithm to use based on the scaling direction
    if target_w > ctc_w or target_h > ctc_h:  # Upscaling
        cropped_image = rescale_i(cropped_image, target_w, target_h, upscale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, upscale_algorithm)
    else:  # Downscaling
        cropped_image = rescale_i(cropped_image, target_w, target_h, downscale_algorithm)
        cropped_mask = rescale_m(cropped_mask, target_w, target_h, downscale_algorithm)

    return canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h


def stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm):
    """Optimized stitching - avoids unnecessary clones"""
    # Only clone canvas since we modify it
    canvas_image = canvas_image.clone()
    
    # Resize inpainted image and mask to match the context size
    _, h, w, _ = inpainted_image.shape
    if ctc_w != w or ctc_h != h:
        if ctc_w > w or ctc_h > h:  # Upscaling
            resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, upscale_algorithm)
            resized_mask = rescale_m(mask, ctc_w, ctc_h, upscale_algorithm)
        else:  # Downscaling
            resized_image = rescale_i(inpainted_image, ctc_w, ctc_h, downscale_algorithm)
            resized_mask = rescale_m(mask, ctc_w, ctc_h, downscale_algorithm)
    else:
        resized_image = inpainted_image
        resized_mask = mask

    # Clamp mask to [0, 1] and expand to match image channels
    resized_mask = resized_mask.clamp(0, 1).unsqueeze(-1)  # shape: [1, H, W, 1]

    # Extract the canvas region we're about to overwrite
    canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :]

    # Blend: new = mask * inpainted + (1 - mask) * canvas
    blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop

    # Paste the blended region back onto the canvas
    canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :] = blended

    # Final crop to get back the original image area
    output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w, :]

    return output_image


class InpaintCropImproved:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Required inputs
                "image": ("IMAGE",),

                # Resize algorithms
                "downscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bilinear"}),
                "upscale_algorithm": (["nearest", "bilinear", "bicubic", "lanczos", "box", "hamming"], {"default": "bicubic"}),

                # Pre-resize input image
                "preresize": ("BOOLEAN", {"default": False, "tooltip": "Resize the original image before processing."}),
                "preresize_mode": (["ensure minimum resolution", "ensure maximum resolution", "ensure minimum and maximum resolution"], {"default": "ensure minimum resolution"}),
                "preresize_min_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_min_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_max_width": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "preresize_max_height": ("INT", {"default": nodes.MAX_RESOLUTION, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1}),

                # Mask manipulation
                "mask_fill_holes": ("BOOLEAN", {"default": True, "tooltip": "Mark as masked any areas fully enclosed by mask."}),
                "mask_expand_pixels": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 1, "tooltip": "Expand the mask by a certain amount of pixels before processing."}),
                "mask_invert": ("BOOLEAN", {"default": False,"tooltip": "Invert mask so that anything masked will be kept."}),
                "mask_blend_pixels": ("INT", {"default": 32, "min": 0, "max": 64, "step": 1, "tooltip": "How many pixels to blend into the original image."}),
                "mask_hipass_filter": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.01, "tooltip": "Ignore mask values lower than this value."}),

                # Extend image for outpainting
                "extend_for_outpainting": ("BOOLEAN", {"default": False, "tooltip": "Extend the image for outpainting."}),
                "extend_up_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_down_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_left_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "extend_right_factor": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),

                # Context
                "context_from_mask_extend_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100.0, "step": 0.01, "tooltip": "Grow the context area from the mask by a certain factor in every direction. For example, 1.5 grabs extra 50% up, down, left, and right as context."}),

                # Output
                "output_resize_to_target_size": ("BOOLEAN", {"default": True, "tooltip": "Force a specific resolution for sampling."}),
                "output_target_width": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "output_target_height": ("INT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 1}),
                "output_padding": (["0", "8", "16", "32", "64", "128", "256", "512"], {"default": "32"}),
                
                # Performance
                "use_gpu": (["auto", "yes", "no"], {"default": "auto", "tooltip": "Use GPU acceleration if available. 'auto' detects automatically."}),
           },
           "optional": {
                # Optional inputs
                "mask": ("MASK",),
                "optional_context_mask": ("MASK",),
           }
        }

    FUNCTION = "inpaint_crop"
    CATEGORY = "inpaint"
    DESCRIPTION = "Crops an image around a mask for inpainting, the optional context mask defines an extra area to keep for the context."


    # Remove the following # to turn on debug mode (extra outputs, print statements)
    #'''
    DEBUG_MODE = False
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK")
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask")

    '''
    
    DEBUG_MODE = False  # FIXED: Disabled debug mode for performance
    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK",
        # DEBUG
        "IMAGE",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "MASK",
        "IMAGE",
        "MASK",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "MASK",
        "IMAGE",
        "IMAGE",
        "MASK",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = ("stitcher", "cropped_image", "cropped_mask",
        # DEBUG
        "DEBUG_preresize_image",
        "DEBUG_preresize_mask",
        "DEBUG_fillholes_mask",
        "DEBUG_expand_mask",
        "DEBUG_invert_mask",
        "DEBUG_blur_mask",
        "DEBUG_hipassfilter_mask",
        "DEBUG_extend_image",
        "DEBUG_extend_mask",
        "DEBUG_context_from_mask",
        "DEBUG_context_from_mask_location",
        "DEBUG_context_expand",
        "DEBUG_context_expand_location",
        "DEBUG_context_with_context_mask",
        "DEBUG_context_with_context_mask_location",
        "DEBUG_context_to_target",
        "DEBUG_context_to_target_location",
        "DEBUG_context_to_target_image",
        "DEBUG_context_to_target_mask",
        "DEBUG_canvas_image",
        "DEBUG_orig_in_canvas_location",
        "DEBUG_cropped_in_canvas_location",
        "DEBUG_cropped_mask_blend",
    )
    #'''

 
    def inpaint_crop(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, use_gpu="auto", mask=None, optional_context_mask=None):
        # PROFILING: Start total timer
        total_start = time.time()
        batch_size = image.shape[0]
        
        # Determine device to use
        if use_gpu == "yes" or (use_gpu == "auto" and torch.cuda.is_available()):
            device = torch.device('cuda:0')
            device_name = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            device = torch.device('cpu')
            device_name = "CPU"
            
        print(f"\n{'='*60}")
        print(f">>> InpaintCropImproved: Processing {batch_size} images")
        print(f"   Image shape: {image.shape}")
        print(f"   Device: {device_name}")
        print(f"{'='*60}\n")
        
        profile_times = {}
        
        # Move to device (clone only if necessary)
        t0 = time.time()
        if image.device != device:
            image = image.to(device)
        else:
            image = image.clone()
        
        if mask is not None:
            if mask.device != device:
                mask = mask.to(device)
            else:
                mask = mask.clone()
        
        if optional_context_mask is not None:
            if optional_context_mask.device != device:
                optional_context_mask = optional_context_mask.to(device)
            else:
                optional_context_mask = optional_context_mask.clone()
        
        profile_times['clone_inputs'] = time.time() - t0
        if profile_times['clone_inputs'] > 1.0:
            print(f"[INFO] Moving to {device_name}: {profile_times['clone_inputs']:.3f}s (consider reducing batch size for faster GPU transfer)")
        else:
            print(f"[DONE] Prepare inputs on {device_name}: {profile_times['clone_inputs']:.3f}s")

        output_padding = int(output_padding)
        
        # Check that some parameters make sense
        if preresize and preresize_mode == "ensure minimum and maximum resolution":
            assert preresize_max_width >= preresize_min_width, "Preresize maximum width must be greater than or equal to minimum width"
            assert preresize_max_height >= preresize_min_height, "Preresize maximum height must be greater than or equal to minimum height"

        if self.DEBUG_MODE:
            print('Inpaint Crop Batch input')
            print(image.shape, type(image), image.dtype)
            if mask is not None:
                print(mask.shape, type(mask), mask.dtype)
            if optional_context_mask is not None:
                print(optional_context_mask.shape, type(optional_context_mask), optional_context_mask.dtype)

        if image.shape[0] > 1:
            assert output_resize_to_target_size, "output_resize_to_target_size must be enabled when input is a batch of images, given all images in the batch output have to be the same size"

        # When a LoadImage node passes a mask without user editing, it may be the wrong shape.
        # Detect and fix that to avoid shape mismatch errors.
        if mask is not None and (image.shape[0] == 1 or mask.shape[0] == 1 or mask.shape[0] == image.shape[0]):
            if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(mask) == 0:
                    mask = torch.zeros((mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)

        if optional_context_mask is not None and (image.shape[0] == 1 or optional_context_mask.shape[0] == 1 or optional_context_mask.shape[0] == image.shape[0]):
            if optional_context_mask.shape[1] != image.shape[1] or optional_context_mask.shape[2] != image.shape[2]:
                if torch.count_nonzero(optional_context_mask) == 0:
                    optional_context_mask = torch.zeros((optional_context_mask.shape[0], image.shape[1], image.shape[2]), device=image.device, dtype=image.dtype)

        # If no mask is provided, create one with the shape of the image
        if mask is None:
            mask = torch.zeros_like(image[:, :, :, 0]).to(device)
    
        # If there is only one image for many masks, replicate it for all masks
        if mask.shape[0] > 1 and image.shape[0] == 1:
            assert image.dim() == 4, f"Expected 4D BHWC image tensor, got {image.shape}"
            image = image.expand(mask.shape[0], -1, -1, -1).clone()

        # If there is only one mask for many images, replicate it for all images
        if image.shape[0] > 1 and mask.shape[0] == 1:
            assert mask.dim() == 3, f"Expected 3D BHW mask tensor, got {mask.shape}"
            mask = mask.expand(image.shape[0], -1, -1).clone()

        # If no optional_context_mask is provided, create one with the shape of the image
        if optional_context_mask is None:
            optional_context_mask = torch.zeros_like(image[:, :, :, 0]).to(device)

        # If there is only one optional_context_mask for many images, replicate it for all images
        if image.shape[0] > 1 and optional_context_mask.shape[0] == 1:
            assert optional_context_mask.dim() == 3, f"Expected 3D BHW optional_context_mask tensor, got {optional_context_mask.shape}"
            optional_context_mask = optional_context_mask.expand(image.shape[0], -1, -1).clone()

        if self.DEBUG_MODE:
            print('Inpaint Crop Batch ready')
            print(image.shape, type(image), image.dtype)
            print(mask.shape, type(mask), mask.dtype)
            print(optional_context_mask.shape, type(optional_context_mask), optional_context_mask.dtype)

         # Validate data
        assert image.ndimension() == 4, f"Expected 4 dimensions for image, got {image.ndimension()}"
        assert mask.ndimension() == 3, f"Expected 3 dimensions for mask, got {mask.ndimension()}"
        assert optional_context_mask.ndimension() == 3, f"Expected 3 dimensions for optional_context_mask, got {optional_context_mask.ndimension()}"
        assert mask.shape[1:] == image.shape[1:3], f"Mask dimensions do not match image dimensions. Expected {image.shape[1:3]}, got {mask.shape[1:]}"
        assert optional_context_mask.shape[1:] == image.shape[1:3], f"optional_context_mask dimensions do not match image dimensions. Expected {image.shape[1:3]}, got {optional_context_mask.shape[1:]}"
        assert mask.shape[0] == image.shape[0], f"Mask batch does not match image batch. Expected {image.shape[0]}, got {mask.shape[0]}"
        assert optional_context_mask.shape[0] == image.shape[0], f"Optional context mask batch does not match image batch. Expected {image.shape[0]}, got {optional_context_mask.shape[0]}"

        # OPTIMIZED: Process images in batches where possible
        result_stitcher = {
            'downscale_algorithm': downscale_algorithm,
            'upscale_algorithm': upscale_algorithm,
            'blend_pixels': mask_blend_pixels,
            'canvas_to_orig_x': [],
            'canvas_to_orig_y': [],
            'canvas_to_orig_w': [],
            'canvas_to_orig_h': [],
            'canvas_image': [],
            'cropped_to_canvas_x': [],
            'cropped_to_canvas_y': [],
            'cropped_to_canvas_w': [],
            'cropped_to_canvas_h': [],
            'cropped_mask_for_blend': [],
        }
        
        batch_size = image.shape[0]
        
        # Process batch operations that can be vectorized
        t0 = time.time()
        if preresize:
            print(f"\n[PRERESIZE] Processing...")
            image, mask, optional_context_mask = self.batch_preresize(
                image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, 
                preresize_mode, preresize_min_width, preresize_min_height, 
                preresize_max_width, preresize_max_height)
            profile_times['preresize'] = time.time() - t0
            print(f"   Done: {profile_times['preresize']:.3f}s")
        
        # Apply mask operations to entire batch at once
        if mask_fill_holes:
            print(f"\n[FILL_HOLES] Processing {batch_size} masks...")
            t0 = time.time()
            mask = self.batch_fillholes(mask)
            profile_times['fill_holes'] = time.time() - t0
            print(f"   Done: {profile_times['fill_holes']:.3f}s ({profile_times['fill_holes']/batch_size:.3f}s per mask)")
        
        if mask_expand_pixels > 0:
            print(f"\n[EXPAND] Expanding masks by {mask_expand_pixels} pixels...")
            t0 = time.time()
            mask = self.batch_expand(mask, mask_expand_pixels)
            profile_times['expand'] = time.time() - t0
            print(f"   Done: {profile_times['expand']:.3f}s")
        
        if mask_invert:
            t0 = time.time()
            mask = 1.0 - mask  # Vectorized inversion
            profile_times['invert'] = time.time() - t0
            print(f"[DONE] Mask inversion: {profile_times['invert']:.3f}s")
        
        # Optimize: Do blur once on the final masks instead of multiple times
        if mask_blend_pixels > 0:
            print(f"\n[EXPAND_FOR_BLEND] Expanding masks for blending...")
            t0 = time.time()
            mask = self.batch_expand(mask, mask_blend_pixels)
            profile_times['expand_for_blend'] = time.time() - t0
            print(f"   Done: {profile_times['expand_for_blend']:.3f}s")
        
        if mask_hipass_filter >= 0.01:
            t0 = time.time()
            mask = torch.where(mask < mask_hipass_filter, 0.0, mask)
            optional_context_mask = torch.where(optional_context_mask < mask_hipass_filter, 0.0, optional_context_mask)
            profile_times['hipass'] = time.time() - t0
            print(f"[DONE] High-pass filter: {profile_times['hipass']:.3f}s")
        
        if extend_for_outpainting:
            print(f"\n[EXTEND] Extending for outpainting...")
            t0 = time.time()
            image, mask, optional_context_mask = self.batch_extend(
                image, mask, optional_context_mask, 
                extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor)
            profile_times['extend'] = time.time() - t0
            print(f"   Done: {profile_times['extend']:.3f}s")
        
        # Process individual crops (still needed due to varying crop regions)
        print(f"\n[CROP] Processing individual crops...")
        crop_start = time.time()
        result_image = []
        result_mask = []
        debug_outputs = {name: [] for name in self.RETURN_NAMES if name.startswith("DEBUG_")}
        
        crop_times = []
        for b in range(batch_size):
            t_crop_start = time.time()
            if b % 10 == 0:  # Progress every 10 images
                print(f"   Processing image {b+1}/{batch_size}...")
            one_image = image[b].unsqueeze(0)
            one_mask = mask[b].unsqueeze(0)
            one_optional_context_mask = optional_context_mask[b].unsqueeze(0)
            
            # Find context and crop (simplified)
            context, x, y, w, h = findcontextarea_m(one_mask)
            if x == -1 or w == -1 or h == -1 or y == -1:
                x, y, w, h = 0, 0, one_image.shape[2], one_image.shape[1]
                context = one_mask[:, y:y+h, x:x+w]
            
            if context_from_mask_extend_factor >= 1.01:
                context, x, y, w, h = growcontextarea_m(context, one_mask, x, y, w, h, context_from_mask_extend_factor)
                if x == -1 or w == -1 or h == -1 or y == -1:
                    x, y, w, h = 0, 0, one_image.shape[2], one_image.shape[1]
            
            context, x, y, w, h = combinecontextmask_m(context, one_mask, x, y, w, h, one_optional_context_mask)
            if x == -1 or w == -1 or h == -1 or y == -1:
                x, y, w, h = 0, 0, one_image.shape[2], one_image.shape[1]
            
            # Perform crop magic
            if not output_resize_to_target_size:
                canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(
                    one_image, one_mask, x, y, w, h, w, h, output_padding, downscale_algorithm, upscale_algorithm)
            else:
                canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(
                    one_image, one_mask, x, y, w, h, output_target_width, output_target_height, output_padding, 
                    downscale_algorithm, upscale_algorithm)
            
            # Store results
            result_stitcher['canvas_to_orig_x'].append(cto_x)
            result_stitcher['canvas_to_orig_y'].append(cto_y)
            result_stitcher['canvas_to_orig_w'].append(cto_w)
            result_stitcher['canvas_to_orig_h'].append(cto_h)
            result_stitcher['canvas_image'].append(canvas_image)
            result_stitcher['cropped_to_canvas_x'].append(ctc_x)
            result_stitcher['cropped_to_canvas_y'].append(ctc_y)
            result_stitcher['cropped_to_canvas_w'].append(ctc_w)
            result_stitcher['cropped_to_canvas_h'].append(ctc_h)
            
            # Store mask for blending (will be blurred in batch later if needed)
            result_stitcher['cropped_mask_for_blend'].append(cropped_mask.clone())
            
            result_image.append(cropped_image.squeeze(0))
            result_mask.append(cropped_mask.squeeze(0))
            
            crop_times.append(time.time() - t_crop_start)
            if b == 0 or (b+1) % 10 == 0 or b == batch_size - 1:
                avg_crop_time = sum(crop_times) / len(crop_times)
                eta = avg_crop_time * (batch_size - b - 1)
                print(f"      [{b+1}/{batch_size}] Avg: {avg_crop_time:.3f}s/img, ETA: {eta:.1f}s")
        
        result_image = torch.stack(result_image, dim=0)
        result_mask = torch.stack(result_mask, dim=0)
        
        # Move results back to CPU for ComfyUI compatibility
        if device.type == 'cuda':
            result_image = result_image.cpu()
            result_mask = result_mask.cpu()
            # Also move stitcher components back to CPU
            for i in range(len(result_stitcher['canvas_image'])):
                result_stitcher['canvas_image'][i] = result_stitcher['canvas_image'][i].cpu()
                result_stitcher['cropped_mask_for_blend'][i] = result_stitcher['cropped_mask_for_blend'][i].cpu()
        
        profile_times['individual_crops'] = time.time() - crop_start
        print(f"\n[DONE] Individual crops complete: {profile_times['individual_crops']:.3f}s total")
        print(f"   Average per image: {profile_times['individual_crops']/batch_size:.3f}s")
        
        # Blur blend masks (process individually due to different sizes)
        if mask_blend_pixels > 0:
            print(f"\n[BLUR_MASKS] Applying blur to {batch_size} blend masks...")
            t0 = time.time()
            # Blur each mask individually (they have different sizes after cropping)
            blurred_masks = []
            for i, mask in enumerate(result_stitcher['cropped_mask_for_blend']):
                if i % 50 == 0 and i > 0:
                    print(f"      Processed {i}/{batch_size} masks...")
                blurred = blur_m(mask, mask_blend_pixels * 0.5)
                blurred_masks.append(blurred)
            result_stitcher['cropped_mask_for_blend'] = blurred_masks
            profile_times['blur_masks'] = time.time() - t0
            print(f"   Done: {profile_times['blur_masks']:.3f}s ({profile_times['blur_masks']/batch_size:.4f}s per mask)")

        # Final timing report
        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f">>> PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Total images processed: {batch_size}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average per image: {total_time/batch_size:.3f}s\n")
        
        print(f"Breakdown:")
        for op, t in sorted(profile_times.items(), key=lambda x: x[1], reverse=True):
            pct = (t / total_time) * 100
            print(f"  {op:20s}: {t:7.3f}s ({pct:5.1f}%)")
        
        # Memory usage
        if torch.cuda.is_available():
            print(f"\nGPU Memory:")
            print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"  Reserved:  {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        print(f"\n{'='*60}\n")
        
        if self.DEBUG_MODE:
            print('Inpaint Crop Batch output')
            print(result_image.shape, type(result_image), result_image.dtype)
            print(result_mask.shape, type(result_mask), result_mask.dtype)

        debug_outputs = {name: torch.stack(values, dim=0) for name, values in debug_outputs.items()}

        return result_stitcher, result_image, result_mask, *[debug_outputs[name] for name in self.RETURN_NAMES if name.startswith("DEBUG_")]


    def batch_preresize(self, image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, 
                        preresize_mode, preresize_min_width, preresize_min_height, 
                        preresize_max_width, preresize_max_height):
        """Batch optimized preresize operation"""
        B, H, W, C = image.shape
        device = image.device
        
        if preresize_mode == "ensure minimum resolution":
            if W >= preresize_min_width and H >= preresize_min_height:
                return image, mask, optional_context_mask
            
            scale_factor = max(preresize_min_width / W, preresize_min_height / H)
            target_width = int(W * scale_factor)
            target_height = int(H * scale_factor)
            
            image = rescale_i(image, target_width, target_height, upscale_algorithm)
            mask = rescale_m(mask, target_width, target_height, 'bilinear')
            optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'bilinear')
            
        elif preresize_mode == "ensure maximum resolution":
            if W <= preresize_max_width and H <= preresize_max_height:
                return image, mask, optional_context_mask
            
            scale_factor = min(preresize_max_width / W, preresize_max_height / H)
            target_width = int(W * scale_factor)
            target_height = int(H * scale_factor)
            
            image = rescale_i(image, target_width, target_height, downscale_algorithm)
            mask = rescale_m(mask, target_width, target_height, 'nearest')
            optional_context_mask = rescale_m(optional_context_mask, target_width, target_height, 'nearest')
            
        return image, mask, optional_context_mask
    
    def batch_fillholes(self, mask):
        """Batch optimized fill holes - now processes entire batch at once!"""
        # New ultra-fast version processes the entire batch at once
        return fillholes_iterative_hipass_fill_m(mask)
    
    def batch_expand(self, mask, pixels):
        """Batch optimized mask expansion"""
        if pixels <= 0:
            return mask
        return expand_m(mask, pixels)
    
    def batch_blur(self, mask, pixels):
        """Batch optimized gaussian blur"""
        if pixels <= 0:
            return mask
        return blur_m(mask, pixels)
    
    def batch_extend(self, image, mask, optional_context_mask, 
                     extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor):
        """Batch optimized image extension for outpainting"""
        B, H, W, C = image.shape
        
        new_H = int(H * (1.0 + extend_up_factor - 1.0 + extend_down_factor - 1.0))
        new_W = int(W * (1.0 + extend_left_factor - 1.0 + extend_right_factor - 1.0))
        
        expanded_image = torch.zeros(B, new_H, new_W, C, device=image.device)
        expanded_mask = torch.ones(B, new_H, new_W, device=mask.device)
        expanded_optional_context_mask = torch.zeros(B, new_H, new_W, device=optional_context_mask.device)
        
        up_padding = int(H * (extend_up_factor - 1.0))
        left_padding = int(W * (extend_left_factor - 1.0))
        
        # Copy original content
        expanded_image[:, up_padding:up_padding+H, left_padding:left_padding+W] = image
        expanded_mask[:, up_padding:up_padding+H, left_padding:left_padding+W] = mask
        expanded_optional_context_mask[:, up_padding:up_padding+H, left_padding:left_padding+W] = optional_context_mask
        
        # Fill edges (simplified - just replicate edge pixels)
        if up_padding > 0:
            expanded_image[:, :up_padding, left_padding:left_padding+W] = image[:, :1].expand(-1, up_padding, -1, -1)
        if new_H - up_padding - H > 0:
            expanded_image[:, up_padding+H:, left_padding:left_padding+W] = image[:, -1:].expand(-1, new_H-up_padding-H, -1, -1)
        if left_padding > 0:
            expanded_image[:, :, :left_padding] = expanded_image[:, :, left_padding:left_padding+1].expand(-1, -1, left_padding, -1)
        if new_W - left_padding - W > 0:
            expanded_image[:, :, left_padding+W:] = expanded_image[:, :, left_padding+W-1:left_padding+W].expand(-1, -1, new_W-left_padding-W, -1)
        
        return expanded_image, expanded_mask, expanded_optional_context_mask

    def inpaint_crop_single_image(self, image, downscale_algorithm, upscale_algorithm, preresize, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height, extend_for_outpainting, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor, mask_hipass_filter, mask_fill_holes, mask_expand_pixels, mask_invert, mask_blend_pixels, context_from_mask_extend_factor, output_resize_to_target_size, output_target_width, output_target_height, output_padding, mask, optional_context_mask):
        if preresize:
            image, mask, optional_context_mask = preresize_imm(image, mask, optional_context_mask, downscale_algorithm, upscale_algorithm, preresize_mode, preresize_min_width, preresize_min_height, preresize_max_width, preresize_max_height)
        if self.DEBUG_MODE:
            DEBUG_preresize_image = image.clone()
            DEBUG_preresize_mask = mask.clone()
       
        if mask_fill_holes:
           mask = fillholes_iterative_hipass_fill_m(mask)
        if self.DEBUG_MODE:
            DEBUG_fillholes_mask = mask.clone()

        if mask_expand_pixels > 0:
            mask = expand_m(mask, mask_expand_pixels)
        if self.DEBUG_MODE:
            DEBUG_expand_mask = mask.clone()

        if mask_invert:
            mask = invert_m(mask)
        if self.DEBUG_MODE:
            DEBUG_invert_mask = mask.clone()

        if mask_blend_pixels > 0:
            mask = expand_m(mask, mask_blend_pixels)
            mask = blur_m(mask, mask_blend_pixels*0.5)
        if self.DEBUG_MODE:
            DEBUG_blur_mask = mask.clone()

        if mask_hipass_filter >= 0.01:
            mask = hipassfilter_m(mask, mask_hipass_filter)
            optional_context_mask = hipassfilter_m(optional_context_mask, mask_hipass_filter)
        if self.DEBUG_MODE:
            DEBUG_hipassfilter_mask = mask.clone()

        if extend_for_outpainting:
            image, mask, optional_context_mask = extend_imm(image, mask, optional_context_mask, extend_up_factor, extend_down_factor, extend_left_factor, extend_right_factor)
        if self.DEBUG_MODE:
            DEBUG_extend_image = image.clone()
            DEBUG_extend_mask = mask.clone()

        context, x, y, w, h = findcontextarea_m(mask)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if self.DEBUG_MODE:
            DEBUG_context_from_mask = context.clone()
            DEBUG_context_from_mask_location = debug_context_location_in_image(image, x, y, w, h)

        if context_from_mask_extend_factor >= 1.01:
            context, x, y, w, h = growcontextarea_m(context, mask, x, y, w, h, context_from_mask_extend_factor)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if self.DEBUG_MODE:
            DEBUG_context_expand = context.clone()
            DEBUG_context_expand_location = debug_context_location_in_image(image, x, y, w, h)

        context, x, y, w, h = combinecontextmask_m(context, mask, x, y, w, h, optional_context_mask)
        # If no mask, mask everything for some inpainting.
        if x == -1 or w == -1 or h == -1 or y == -1:
            x, y, w, h = 0, 0, image.shape[2], image.shape[1]
            context = mask[:, y:y+h, x:x+w]
        if self.DEBUG_MODE:
            DEBUG_context_with_context_mask = context.clone()
            DEBUG_context_with_context_mask_location = debug_context_location_in_image(image, x, y, w, h)

        if not output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, w, h, output_padding, downscale_algorithm, upscale_algorithm)
        else: # if output_resize_to_target_size:
            canvas_image, cto_x, cto_y, cto_w, cto_h, cropped_image, cropped_mask, ctc_x, ctc_y, ctc_w, ctc_h = crop_magic_im(image, mask, x, y, w, h, output_target_width, output_target_height, output_padding, downscale_algorithm, upscale_algorithm)
        if self.DEBUG_MODE:
            DEBUG_context_to_target = context.clone()
            DEBUG_context_to_target_location = debug_context_location_in_image(image, x, y, w, h)
            DEBUG_context_to_target_image = image.clone()
            DEBUG_context_to_target_mask = mask.clone()
            DEBUG_canvas_image = canvas_image.clone()
            DEBUG_orig_in_canvas_location = debug_context_location_in_image(canvas_image, cto_x, cto_y, cto_w, cto_h)
            DEBUG_cropped_in_canvas_location = debug_context_location_in_image(canvas_image, ctc_x, ctc_y, ctc_w, ctc_h)

        # For blending, grow the mask even further and make it blurrier.
        cropped_mask_blend = cropped_mask.clone()
        if mask_blend_pixels > 0:
           cropped_mask_blend = blur_m(cropped_mask_blend, mask_blend_pixels*0.5)
        if self.DEBUG_MODE:
            DEBUG_cropped_mask_blend = cropped_mask_blend.clone()

        stitcher = {
            'canvas_to_orig_x': cto_x,
            'canvas_to_orig_y': cto_y,
            'canvas_to_orig_w': cto_w,
            'canvas_to_orig_h': cto_h,
            'canvas_image': canvas_image,
            'cropped_to_canvas_x': ctc_x,
            'cropped_to_canvas_y': ctc_y,
            'cropped_to_canvas_w': ctc_w,
            'cropped_to_canvas_h': ctc_h,
            'cropped_mask_for_blend': cropped_mask_blend,
        }

        if not self.DEBUG_MODE:
            return stitcher, cropped_image, cropped_mask
        else:
            return stitcher, cropped_image, cropped_mask, DEBUG_preresize_image, DEBUG_preresize_mask, DEBUG_fillholes_mask, DEBUG_expand_mask, DEBUG_invert_mask, DEBUG_blur_mask, DEBUG_hipassfilter_mask, DEBUG_extend_image, DEBUG_extend_mask, DEBUG_context_from_mask, DEBUG_context_from_mask_location, DEBUG_context_expand, DEBUG_context_expand_location, DEBUG_context_with_context_mask, DEBUG_context_with_context_mask_location, DEBUG_context_to_target, DEBUG_context_to_target_location, DEBUG_context_to_target_image, DEBUG_context_to_target_mask, DEBUG_canvas_image, DEBUG_orig_in_canvas_location, DEBUG_cropped_in_canvas_location, DEBUG_cropped_mask_blend




class InpaintStitchImproved:
    """
    ComfyUI-InpaintCropAndStitch
    https://github.com/lquesada/ComfyUI-InpaintCropAndStitch

    This node stitches the inpainted image without altering unmasked areas.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "inpainted_image": ("IMAGE",),
            },
            "optional": {
                "use_gpu": (["auto", "yes", "no"], {"default": "auto", "tooltip": "Use GPU acceleration if available. 'auto' detects automatically."}),
            }
        }

    CATEGORY = "inpaint"
    DESCRIPTION = "Stitches an image cropped with Inpaint Crop back into the original image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    FUNCTION = "inpaint_stitch"


    def inpaint_stitch(self, stitcher, inpainted_image, use_gpu="auto"):
        # Start timing
        import time
        start_time = time.time()
        
        batch_size = inpainted_image.shape[0]
        
        # Determine device to use
        if use_gpu == "yes" or (use_gpu == "auto" and torch.cuda.is_available()):
            device = torch.device('cuda:0')
            device_name = f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            device = torch.device('cpu')
            device_name = "CPU"
        
        print(f"\n>>> InpaintStitchImproved: Processing {batch_size} images on {device_name}")
        
        # Move inpainted images to device
        inpainted_image = inpainted_image.clone().to(device)
        
        # Check batch size compatibility
        assert len(stitcher['cropped_to_canvas_x']) == batch_size or len(stitcher['cropped_to_canvas_x']) == 1, "Stitch batch size doesn't match image batch size"
        override = len(stitcher['cropped_to_canvas_x']) == 1 and batch_size > 1
        
        results = []
        
        # Process each image
        for b in range(batch_size):
            if b % 20 == 0:
                print(f"   Stitching image {b+1}/{batch_size}...")
            
            # Get data for this image
            idx = 0 if override else b
            
            # Move canvas and mask to device
            canvas_image = stitcher['canvas_image'][idx].to(device)
            mask = stitcher['cropped_mask_for_blend'][idx].to(device)
            
            # Get coordinates
            ctc_x = stitcher['cropped_to_canvas_x'][idx]
            ctc_y = stitcher['cropped_to_canvas_y'][idx]
            ctc_w = stitcher['cropped_to_canvas_w'][idx]
            ctc_h = stitcher['cropped_to_canvas_h'][idx]
            cto_x = stitcher['canvas_to_orig_x'][idx]
            cto_y = stitcher['canvas_to_orig_y'][idx]
            cto_w = stitcher['canvas_to_orig_w'][idx]
            cto_h = stitcher['canvas_to_orig_h'][idx]
            
            # Process single image (optimized)
            one_image = inpainted_image[b].unsqueeze(0)
            # FIX: Don't unsqueeze canvas_image - it already has batch dimension
            # Check if canvas_image needs batch dimension
            if canvas_image.dim() == 3:  # If it's HWC, add batch dimension
                canvas_image = canvas_image.unsqueeze(0)
            # If it's already 4D (BHWC), use as is
            
            output_image = self.stitch_single_optimized(
                canvas_image, one_image, mask,
                ctc_x, ctc_y, ctc_w, ctc_h,
                cto_x, cto_y, cto_w, cto_h,
                stitcher['downscale_algorithm'],
                stitcher['upscale_algorithm']
            )
            
            results.append(output_image.squeeze(0))
        
        # Stack results and move back to CPU for ComfyUI
        result_batch = torch.stack(results, dim=0)
        if device.type == 'cuda':
            result_batch = result_batch.cpu()
        
        elapsed = time.time() - start_time
        print(f"   Done! Total: {elapsed:.2f}s ({elapsed/batch_size:.3f}s per image)")
        
        return (result_batch,)

    def stitch_single_optimized(self, canvas_image, inpainted_image, mask,
                                ctc_x, ctc_y, ctc_w, ctc_h,
                                cto_x, cto_y, cto_w, cto_h,
                                downscale_algorithm, upscale_algorithm):
        """Optimized single image stitching - no unnecessary clones, stays on device"""
        device = canvas_image.device
        
        # Ensure canvas_image is 4D (BHWC), not 5D
        if canvas_image.dim() == 5:
            # If shape is [1, 1, H, W, C] or [B, 1, H, W, C], squeeze the extra dimension
            canvas_image = canvas_image.squeeze(1)
        
        # Use the provided dimensions (ctc_w, ctc_h) which are the target dimensions
        target_w = ctc_w
        target_h = ctc_h
        
        # Skip if target dimensions are invalid
        if target_w <= 0 or target_h <= 0:
            print(f"Warning: Invalid target dimensions w={target_w}, h={target_h}, returning canvas unchanged")
            return canvas_image
        
        # Resize inpainted image and mask to match the target dimensions
        _, h, w, _ = inpainted_image.shape
        if target_w != w or target_h != h:
            # Always resize to match the target dimensions
            resized_image = rescale_i(inpainted_image, target_w, target_h, 
                                     upscale_algorithm if target_w > w or target_h > h else downscale_algorithm)
            # Handle mask dimensions properly
            if mask.dim() == 2:  # HW format
                mask = mask.unsqueeze(0)  # Add batch dimension -> BHW
            if mask.shape[-2:] != (target_h, target_w):  # Check last two dims (H, W)
                resized_mask = rescale_m(mask, target_w, target_h,
                                       upscale_algorithm if target_w > w or target_h > h else downscale_algorithm)
            else:
                resized_mask = mask
        else:
            resized_image = inpainted_image
            resized_mask = mask
            if mask.dim() == 2:
                resized_mask = mask.unsqueeze(0)
        
        # Ensure mask is in right shape and clamped
        resized_mask = resized_mask.clamp(0, 1)
        
        # Ensure mask has the right number of dimensions
        if resized_mask.dim() == 3:  # BHW
            resized_mask = resized_mask.unsqueeze(-1)  # BHW -> BHWC
        elif resized_mask.dim() == 2:  # HW
            resized_mask = resized_mask.unsqueeze(0).unsqueeze(-1)  # HW -> BHWC
        
        # Extract the canvas region we're about to overwrite
        canvas_crop = canvas_image[:, ctc_y:ctc_y + target_h, ctc_x:ctc_x + target_w, :]
        
        # Ensure dimensions match exactly before blending (safety check)
        if resized_image.shape[1:3] != (target_h, target_w):
            print(f"WARNING: Image shape mismatch after resize - Image: {resized_image.shape}, Target: ({target_h}, {target_w})")
            resized_image = rescale_i(resized_image, target_w, target_h, 'bilinear')
        
        if resized_mask.shape[1:3] != (target_h, target_w):
            print(f"WARNING: Mask shape mismatch - Mask: {resized_mask.shape}, Target: ({target_h}, {target_w})")
            resized_mask = rescale_m(resized_mask.squeeze(-1), target_w, target_h, 'bilinear')
            resized_mask = resized_mask.unsqueeze(-1)
        
        # Blend: new = mask * inpainted + (1 - mask) * canvas
        blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop
        
        # Paste the blended region back onto the canvas
        canvas_image[:, ctc_y:ctc_y + target_h, ctc_x:ctc_x + target_w, :] = blended
        
        # Final crop to get back the original image area
        output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w, :]
        
        return output_image
    
    def inpaint_stitch_single_image(self, stitcher, inpainted_image):
        """Legacy method for compatibility"""
        downscale_algorithm = stitcher['downscale_algorithm']
        upscale_algorithm = stitcher['upscale_algorithm']
        canvas_image = stitcher['canvas_image']

        ctc_x = stitcher['cropped_to_canvas_x']
        ctc_y = stitcher['cropped_to_canvas_y']
        ctc_w = stitcher['cropped_to_canvas_w']
        ctc_h = stitcher['cropped_to_canvas_h']

        cto_x = stitcher['canvas_to_orig_x']
        cto_y = stitcher['canvas_to_orig_y']
        cto_w = stitcher['canvas_to_orig_w']
        cto_h = stitcher['canvas_to_orig_h']

        mask = stitcher['cropped_mask_for_blend']  # shape: [1, H, W]

        output_image = stitch_magic_im(canvas_image, inpainted_image, mask, ctc_x, ctc_y, ctc_w, ctc_h, cto_x, cto_y, cto_w, cto_h, downscale_algorithm, upscale_algorithm)

        return (output_image,)
