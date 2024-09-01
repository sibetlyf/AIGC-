from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto import Random
from Crypto.PublicKey import RSA
import base64
import os
import sys
import random
current_path = os.path.dirname(os.path.abspath(__file__))

# 生成随机字符串
def generate_random_str(randomlength=16):
    """
    生成一个指定长度的随机字符串
    """
    random_str =''
    base_str ='ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
    length =len(base_str) -1
    for i in range(randomlength):
        random_str +=base_str[random.randint(0, length)]

    print(str)
    return random_str


# RSA非对称加密传输密钥
## 生成公私钥
def generate_rsa_key():
    random_generator = Random.new().read
    key = RSA.generate(2048, random_generator)
    private_key = key.export_key()
    print(private_key.decode('utf-8'))
    public_key = key.publickey().export_key()
    print(public_key.decode('utf-8'))
    
    # 写入文件中
    key_path = os.path.join(current_path, 'encryptkeys')
    os.makedirs(key_path, exist_ok=True)
    with open(os.path.join(key_path, 'private_key.pem'), 'wb') as f:
        f.write(private_key)
    with open(os.path.join(key_path, 'public_key.pem'), 'wb') as f:
        f.write(public_key)

## 公钥加密，私钥解密
import base64
from Crypto.PublicKey import RSA
from Crypto.Hash import SHA
from Crypto.Signature import PKCS1_v1_5 as PKCS1_signature
from Crypto.Cipher import PKCS1_v1_5 as PKCS1_cipher

class RSACipher:
    def __init__(self, key_file=os.path.join(current_path, 'encryptkeys')):
        self.key_file = key_file
        self.public_key = self.get_key(os.path.join(key_file, 'public_key.pem'))
        self.private_key = self.get_key(os.path.join(key_file, 'private_key.pem'))

    def get_key(self, key_file):
        with open(key_file) as f:
            data = f.read()
            key = RSA.importKey(data)

        return key
    
    def encrypt_data(self, msg):
        public_key = self.public_key
        cipher = PKCS1_cipher.new(public_key)
        encrypt_text = base64.b64encode(cipher.encrypt(bytes(msg.encode("utf-8"))))
        return encrypt_text.decode('utf-8')

    def decrypt_data(self, encrypt_msg):
        private_key = self.private_key
        cipher = PKCS1_cipher.new(private_key)
        back_text = cipher.decrypt(base64.b64decode(encrypt_msg), 0)
        return back_text.decode('utf-8')

def test_encrypt_decrypt():
    msg = "你好，这是一个测试！"
    rsa = RSACipher()
    encrypt_text = rsa.encrypt_data(msg)
    decrypt_text = rsa.decrypt_data(encrypt_text)
    print(msg == decrypt_text)



# AES对称加密内容
class AESCipher:
    def __init__(self, key="Sixteen byte key"):
        self.key = key.encode('utf-8')  # 确保key是字节串
        self.cipher = AES.new(self.key, AES.MODE_ECB)

    def encrypt(self, plaintext):
        """加密给定的文本，并打印加密信息"""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')  # 将字符串转换为字节串
        print(f"正在加密: {plaintext.decode('utf-8')}")
        padded_plaintext = pad(plaintext, AES.block_size)
        encrypted_data = self.cipher.encrypt(padded_plaintext)
        return base64.b64encode(encrypted_data).decode('utf-8')

    def decrypt(self, ciphertext):
        """解密给定的密文，并打印解密信息"""
        print(f"正在解密: {ciphertext}")
        decrypted_data = self.cipher.decrypt(base64.b64decode(ciphertext))
        unpadded_data = unpad(decrypted_data, AES.block_size)
        # 使用UTF-8解码
        utf8_decoded = unpadded_data.decode('utf-8')

        return utf8_decoded

class Encrypter():
    def __init__(self, key="Sixteen byte key", key_file=os.path.join(current_path, 'encryptkeys')):
        self.rsa = RSACipher()
        # AES加密内容
        self.aes = AESCipher(key)
        # RSA密钥地址
        self.key_file = key_file
    
    def send_encrypted_message(self, message):
        # 生成随机AES密钥
        aes_key = generate_random_str(randomlength=16)
        # 使用RSA公钥加密AES密钥
        encrypted_aes_key = self.rsa.encrypt_data(aes_key)
        # 使用AES密钥加密消息
        enerypted_message = self.aes.encrypt(message)

        return encrypted_aes_key, enerypted_message

    def receive_and_decrypt_message(self, encrypted_aes_key, encrypted_message):
        # 使用RSA私钥解密AES密钥
        aes_key = self.rsa.decrypt_data(encrypted_aes_key)
        # 使用AES密钥解密消息
        decrypted_messages = self.aes.decrypt(encrypted_message)
        
        return decrypted_messages


# 使用示例
if __name__ == '__main__':
    # key = 'Sixteen byte key'
    # cipher = AESCipher(key)

    # # 加密
    # plaintext = '你好，这是一个测试！'
    # encrypted_text = cipher.encrypt(plaintext)
    # print(f'Encrypted: {encrypted_text}')

    # # 解密
    # decrypted_text = cipher.decrypt(encrypted_text)
    # print(f'Decrypted: {decrypted_text}')
    
    # # RSA
    # generate_rsa_key()
    # test_encrypt_decrypt()
    
    # 生成RSA密钥对
    generate_rsa_key()

    # 创建Encrypter实例
    encrypter = Encrypter()

    # 原始消息
    message = '你好，这是一个测试！'

    # 发送加密消息
    encrypted_aes_key, encrypted_message = encrypter.send_encrypted_message(message)
    print(f'Encrypted AES Key: {encrypted_aes_key}')
    print(f'Encrypted Message: {encrypted_message}')

    # 接收并解密消息
    decrypted_message = encrypter.receive_and_decrypt_message(encrypted_aes_key, encrypted_message)
    print(f'Decrypted Message: {decrypted_message}')