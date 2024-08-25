import sys
import os
import pandas as pd
import torch
from training.train import train_model

# Thêm thư mục gốc của dự án vào sys.path để Python có thể tìm thấy module src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    # Đọc file CSV
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'TrainningSet_161.csv')
    data = pd.read_csv(data_path)
    
    # Huấn luyện mô hình
    trained_model = train_model(data, epochs=10, batch_size=8, lr=1e-3)

    # Tạo thư mục để lưu mô hình nếu chưa tồn tại
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_models')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Lưu mô hình
    model_save_path = os.path.join(save_dir, 'model.pth')
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Training complete. Model saved as '{model_save_path}'.")
