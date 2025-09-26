# import nltk
# import os

# Tạo thư mục nếu chưa có
# nltk_data_dir = os.path.expanduser('./nltk_data')
# os.makedirs(nltk_data_dir, exist_ok=True)

# # Thêm đường dẫn vào cấu hình của NLTK
# nltk.data.path.append(nltk_data_dir)

# # Tải tài nguyên
# nltk.download('punkt', download_dir=nltk_data_dir)
# nltk.download('wordnet', download_dir=nltk_data_dir)
# nltk.download('omw-1.4', download_dir=nltk_data_dir)  # rất quan trọng cho WordNetLemmatizer mới
# nltk.download('stopwords', download_dir=nltk_data_dir)

import nltk
nltk.download('wordnet', download_dir='C:\\Users\\LENOVO\\nltk_data')
nltk.data.path.append('C:\\Users\\LENOVO\\nltk_data')
