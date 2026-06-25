# Báo cáo Tiến độ Dự án: Spam Detection MLOps Pipeline

Dưới đây là bảng tổng hợp các bước đã hoàn thiện  và các bước còn lại cần thực hiện trong thời gian tới để hoàn tất dự án cuối môn.

---

## 1. Các bước ĐÃ HOÀN THÀNH (Completed)

 Nội dung công việc | Kết quả đạt được | Trạng thái |
| :---: | :--- | :--- | :---: |
 -Thiết lập môi trường & cấu trúc thư mục | Khởi tạo thư mục dự án tại `D:/spam_detection`, tạo môi trường ảo `.venv`, viết `.gitignore` và `requirements.txt`. | **Đã xong** ✅ |
 -Thu thập và xử lý tập dữ liệu | Viết script [download_data.py](src/download_data.py) tải dữ liệu SMS Spam từ UCI và chuyển đổi thành [spam.csv](data/spam.csv) (5.572 dòng). | **Đã xong** ✅ |
 -Tiền xử lý & Huấn luyện mô hình | Viết [train.py](src/train.py) xây dựng pipeline TF-IDF + Multinomial Naive Bayes. Lưu mô hình thành `model.joblib` và `vectorizer.joblib`. | **Đã xong** ✅ |
 -Đánh giá mô hình & Lưu metrics | Viết [evaluate.py](src/evaluate.py) đo lường các chỉ số và lưu thành [metrics.json](models/metrics.json). Độ chính xác đạt **97.04%**, Precision đạt **100%**. | **Đã xong** ✅ |
 -Xây dựng logic suy luận (Inference) | Viết [predict.py](src/predict.py) hỗ trợ chạy dự đoán tin nhắn Ham/Spam trực tiếp qua giao diện dòng lệnh (CLI). | **Đã xong** ✅ |
 -Xây dựng Web API bằng FastAPI | Viết [app.py](src/app.py) tạo web service với endpoint `POST /predict` để nhận request JSON và trả về kết quả dự đoán kèm xác suất. | **Đã xong** ✅ |
 -Viết Unit Tests tự động | Viết 7 test cases tại thư mục [tests/](tests/) và chạy thành công 100% bằng thư viện `pytest`. | **Đã xong** ✅ |
 -Docker hóa ứng dụng | Viết thành công [Dockerfile](Dockerfile) và [.dockerignore](.dockerignore) để đóng gói dịch vụ FastAPI. | **Đã xong** ✅ |
 -Build và chạy thử nghiệm Docker | Build thành công image `spam-api:latest`, chạy thử container và test gọi API hoạt động ổn định trên cổng `8000`. | **Đã xong** ✅ |
 -Thiết lập CI Workflow | Tạo tệp cấu hình [.github/workflows/ci.yml](.github/workflows/ci.yml) để tự động hóa kiểm thử và train mô hình mỗi khi push code lên GitHub. | **Đã xong** ✅ |

---

## 2. Các bước CHƯA LÀM (Pending)

Dưới đây là các phần việc còn lại của Tuần 2 tập trung vào việc triển khai CD (Continuous Delivery/Deployment):

*   - Cấu hình và kết nối GitHub Self-hosted Runner**
    *   *Nhiệm vụ*: Đăng ký và kết nối máy tính local của bạn làm Runner trên repo GitHub để sẵn sàng nhận lệnh deploy.
*   - Thiết lập CD workflow cơ bản (`cd.yml`)**
    *   *Nhiệm vụ*: Viết file cấu hình `.github/workflows/cd.yml` chạy trên môi trường `self-hosted` để tự động build Docker image mới khi đẩy code lên branch `main`.
*   - Hoàn thiện CD (Tự động hóa restart container)**
    *   *Nhiệm vụ*: Cấu hình CD để tự động dừng container cũ đang chạy, xóa container cũ và khởi chạy container mới với code cập nhật để tránh xung đột cổng `8000`.
*   - Viết tài liệu hướng dẫn và chạy thử kiểm thử đầu-cuối**
    *   *Nhiệm vụ*: Viết tài liệu hướng dẫn sử dụng chi tiết tại [README.md](README.md), chạy thử toàn bộ pipeline tự động từ push code đến deploy và đóng gói dự án để báo cáo.

---

