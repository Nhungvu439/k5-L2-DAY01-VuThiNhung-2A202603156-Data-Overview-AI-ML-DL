# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** Chưa xác định CPU/GPU

**Python / PyTorch / Ultralytics:** Ultralytics 8.4.145; Python và PyTorch chưa ghi nhận.

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không ghi nhận thay đổi.

> **Lưu ý:** JSON Detection được cung cấp có `sample_id = traffic`, trong khi ảnh minh họa Detection và Segmentation hiển thị ảnh `kitchen`. Cần kiểm tra lại dữ liệu trước khi nộp.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K.

2. Record này nói về **cả bức ảnh**, không phải một vật thể riêng. Vì Image Classification phân loại nội dung tổng thể của ảnh, không xác định vị trí của từng vật thể.

3. Danh sách tên lớp được lấy từ taxonomy đi kèm model/checkpoint, không phải do model tự nghĩ ra. Trong bài này sử dụng taxonomy ImageNet-1K.

4. Cần ghi cả `class_id`, `class_name` và `taxonomy_name` vì:
   - `class_id` là mã số định danh của lớp.
   - `class_name` là tên lớp giúp con người dễ hiểu.
   - `taxonomy_name` cho biết danh sách lớp đang được sử dụng.

   Ba thông tin này giúp tránh nhầm lẫn giữa các bộ dữ liệu khác nhau.

5. Nếu ảnh có nhiều vật hoặc nhiều chủ đề, guideline cần quy định rõ cách xác định chủ thể chính, tiêu chí chọn lớp đại diện cho toàn bộ ảnh và cách xử lý khi có nhiều đối tượng nổi bật.

6. Không được lấy `score` của model làm ground truth vì score chỉ thể hiện mức độ tin cậy của model đối với dự đoán, không chứng minh dự đoán đó chính xác. Ground truth phải do con người xác định theo guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
JSON được cung cấp có `sample_id = traffic`. Chọn record bus có score cao nhất trong dữ liệu đã gửi.

1. Chép một record: `class_name`=bus ; `score`=0.912558 ; `bbox_xyxy`=[93.17, 187.95, 223.01, 320.91] ; `bbox_width`=129.84 ; `bbox_height`=132.96.

2. Giải thích box bằng lời thường:
   - Góc trên-trái ở tọa độ (93.17, 187.95).
   - Góc dưới-phải ở tọa độ (223.01, 320.91).
   - Chiều rộng: 129.84 pixel.
   - Chiều cao: 132.96 pixel.

   Tọa độ được tính từ góc trên bên trái của ảnh.

3. Số lượng prediction theo ngưỡng:
   - Ngưỡng 0.20: Chưa có dữ liệu.
   - Ngưỡng 0.35: 53 prediction.
   - Ngưỡng 0.60: Chưa có dữ liệu.

4. Khi hạ ngưỡng, model giữ lại nhiều hộp dự đoán hơn, bao gồm cả những dự đoán có độ tin cậy thấp. Reviewer cần kiểm tra kỹ hơn các hộp nhận diện sai, hộp trùng lặp, vật thể bị nhận diện nhầm và những trường hợp phát hiện không chính xác.

5. Một quy tắc box: Người gán nhãn phải vẽ hình chữ nhật bao quanh toàn bộ phần vật thể nhìn thấy được, sát với ranh giới vật thể nhất có thể và không lấy quá nhiều phần nền xung quanh.

6. Trên ảnh kitchen, có thể quan sát thấy một số vật thể trong khu vực bếp bị che khuất hoặc nằm gần nhau. Với những trường hợp không xác định rõ ranh giới, annotator nên escalation lên Lab Coach thay vì tự phỏng đoán. Nếu trường hợp đã có quy tắc rõ ràng trong guideline thì thực hiện theo quy tắc đó.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Ảnh minh họa được cung cấp là ảnh `kitchen`. Chưa có nội dung JSON Segmentation đầy đủ để xác nhận instance và polygon cụ thể.

1. Chép một record:
   - `instance_id`=Chưa có dữ liệu JSON
   - `class_name`=person (theo ảnh minh họa)
   - `score`=0.90 (theo nhãn hiển thị trên ảnh)
   - Số điểm polygon (`polygon_point_count`)=Chưa có dữ liệu JSON
   - 3–5 điểm đầu của `polygon_xy`=Chưa có dữ liệu JSON

2. So với box, polygon cung cấp thêm thông tin chi tiết về hình dạng và đường biên thực tế của vật thể. Bounding box chỉ là hình chữ nhật bao quanh vật thể, còn polygon có thể mô tả đường viền không đều và chính xác hơn.

3. `instance_id` dùng để phân biệt từng đối tượng riêng biệt trong cùng một ảnh. Nó không phải class ID vì class ID xác định loại đối tượng, còn instance ID xác định một đối tượng cụ thể. Nó cũng không phải tracking ID vì tracking ID dùng để theo dõi một đối tượng qua nhiều khung hình video.

4. Một quy tắc vẽ biên mask: Người gán nhãn cần vẽ đường biên theo ranh giới thực tế của phần vật thể nhìn thấy được, không bao gồm phần nền hoặc vật thể khác. Đường biên cần sát với hình dạng đối tượng và nhất quán giữa các ảnh.

5. Trên ảnh kitchen, có thể quan sát thấy các vật thể trong khu vực bếp nằm gần nhau và một số đường biên bị che khuất. Annotator cần tuân thủ guideline đã thống nhất. Nếu không thể xác định chính xác đường biên giữa hai vật thể thì cần escalation lên Lab Coach thay vì tự phỏng đoán.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một class label đại diện cho toàn bộ ảnh. | Ảnh có nhiều loại phương tiện nên việc chọn một lớp đại diện có thể gây nhầm lẫn. Model dự đoán Top 1 là `cab` với score 0.510915. | Chọn lớp phù hợp nhất với nội dung tổng thể của ảnh theo guideline. | Kiểm tra lớp được chọn có đại diện đúng cho toàn bộ ảnh hay không. |
| Phát hiện vật thể | Class label và bounding box cho từng vật thể. | Có nhiều vật thể, một số vật thể nhỏ, bị che khuất hoặc nằm sát mép ảnh. JSON ghi nhận 53 prediction tại threshold 0.35. | Vẽ bounding box chính xác, gắn đúng class cho từng vật thể nhìn thấy. | Kiểm tra class, vị trí và kích thước box; phát hiện vật thể bị bỏ sót, box sai hoặc trùng lặp. |
| Instance segmentation | Class label và polygon mask riêng cho từng đối tượng. | Đường biên vật thể có thể không rõ, các vật thể có thể chạm nhau hoặc bị che khuất. | Vẽ mask theo đường viền thực tế của từng đối tượng. | Kiểm tra độ chính xác của mask, đường biên polygon và việc phân biệt các instance. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của tôi là: Chỉ sử dụng dữ liệu mẫu được cung cấp cho bài lab, không tải lên hoặc chia sẻ dữ liệu cá nhân, hình ảnh riêng tư hay dữ liệu nhạy cảm lên các nền tảng công khai.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng xử lý và báo cho Lab Coach để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
