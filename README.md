# AI-Powered Stock Analyzing & Forecasting

Ứng dụng này sử dụng AI và các mô hình học sâu để phân tích, dự báo giá cổ phiếu, tiền điện tử, chỉ số thế giới, vàng và ngoại tệ. Giao diện trực quan với Streamlit, tích hợp các chỉ báo kỹ thuật, XAI (giải thích mô hình), và phân tích tài chính tự động bằng AI.

## Tính năng nổi bật

- **Phân tích kỹ thuật**: SMA, EMA, Bollinger Bands, RSI, MACD, OBV, ADX, Stochastic Oscillator, VWAP...
- **Dự báo giá**: ARIMA, LSTM, Transformer (tự huấn luyện trên dữ liệu thực tế).
- **XAI**: Giải thích dự báo bằng SHAP, LIME, Attention (Transformer).
- **Phân tích tài chính & cảm xúc**: Tự động gửi báo cáo tài chính và tin tức cho AI phân tích, đưa ra khuyến nghị Mua/Bán/Giữ.
- **Hỗ trợ đa thị trường**: Cổ phiếu Việt Nam, tiền điện tử, chỉ số thế giới, giá vàng, tỷ giá ngoại tệ.
- **Tải dữ liệu dự báo**: Xuất file CSV kết quả dự báo.
- **Tích hợp Ollama API**: Phân tích tài chính và giải thích XAI bằng mô hình AI lớn.

## Yêu cầu hệ thống

- Python 3.10+
- Windows (khuyến nghị)
- Đã cài đặt [Ollama](https://ollama.com/) và chạy API tại `http://localhost:11434`
- Trình duyệt Edge (cho tính năng chụp ảnh biểu đồ)

## Cài đặt

1. **Clone dự án:**
    ```
    git clone https://github.com/ntdung2212/ProjectStock.git
    cd ProjectStock
    ```

2. **Cài đặt thư viện:**
    ```
    pip install -r requirements.txt
    ```

3. **Chạy ứng dụng:**
    ```
    streamlit run app.py
    ```

## Hướng dẫn sử dụng

- Chọn chức năng ở thanh bên: Phân tích cổ phiếu, Dự báo, Vàng & Ngoại tệ, Crypto, World Index.
- Chọn mã cổ phiếu, thời gian, chỉ báo kỹ thuật, mô hình dự báo.
- Có thể huấn luyện lại mô hình LSTM/Transformer với dữ liệu mới.
- Xem giải thích mô hình (XAI) và phân tích tài chính tự động bằng AI.
- Tải kết quả dự báo về máy.

## Thư viện sử dụng

- `streamlit`, `pandas`, `plotly`, `matplotlib`, `seaborn`, `numpy`
- `tensorflow`, `keras`, `pmdarima`, `scikit-learn`
- `shap`, `lime`, `selenium`, `webdriver-manager`
- `vnstock` (lấy dữ liệu tài chính Việt Nam)
- `requests`, `base64`, `os`, `time`

## Lưu ý

- Ứng dụng chỉ mang tính chất tham khảo, không phải là lời khuyên đầu tư.
- Để sử dụng phân tích AI, cần chạy Ollama API với mô hình tương thích (ví dụ: gemma3).
- Một số tính năng (chụp ảnh biểu đồ) yêu cầu Edge WebDriver.

---
