from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

def performance_evalutation(image,deconvolved_image):
    mse = mean_squared_error(image, deconvolved_image)
    psnr = peak_signal_noise_ratio(image, deconvolved_image)
    ssim = structural_similarity(image, deconvolved_image)

    # Afficher les métriques
    print(f"MSE : {mse:.4f}")
    print(f"PSNR : {psnr:.4f}")
    print(f"SSIM : {ssim:.4f}")
    return mse,psnr,ssim