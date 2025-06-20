import base64

def base64_to_image(base64_string, output_path):
    """
    将Base64编码的字符串转换为图片并保存到本地。

    :param base64_string: Base64编码的字符串（不包含头部信息）
    :param output_path: 图片保存的路径（包括文件名和扩展名）
    """
    try:
        # 去掉可能存在的头部信息（如"data:image/png;base64,"）
        if ',' in base64_string:
            header, base64_data = base64_string.split(',', 1)
        else:
            base64_data = base64_string

        # 解码Base64字符串
        image_data = base64.b64decode(base64_data)

        # 将解码后的数据写入文件
        with open(output_path, 'wb') as image_file:
            image_file.write(image_data)

        print(f"图片已成功保存到 {output_path}")
    except Exception as e:
        print(f"发生错误：{e}")