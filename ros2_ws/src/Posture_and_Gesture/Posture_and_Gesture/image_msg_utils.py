import cv2

from sensor_msgs.msg import CompressedImage


def bgr8_to_jpeg_compressed_image(header, bgr_image, jpeg_quality):
    jpeg_quality = max(1, min(100, int(jpeg_quality)))
    success, encoded = cv2.imencode(
        '.jpg',
        bgr_image,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not success:
        return None

    image_msg = CompressedImage()
    image_msg.header = header
    image_msg.format = 'bgr8; jpeg compressed bgr8'
    image_msg.data = encoded.tobytes()
    return image_msg
