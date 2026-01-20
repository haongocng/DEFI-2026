import re

def find_missing_samples(result_file, total_expected=932):
    print(f"--- Đang kiểm tra file: {result_file} ---")
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Không tìm thấy file {result_file}")
        return

    # 1. Tìm tất cả các tiêu đề SAMPLE X để biết ID nào đã xuất hiện
    # Tìm các định dạng: "--- SAMPLE 1 ---" hoặc "SAMPLE 1:"
    found_ids = set()
    id_matches = re.findall(r'(?:SAMPLE|--- SAMPLE)\s+(\d+)', content, re.IGNORECASE)
    
    for id_str in id_matches:
        found_ids.add(int(id_str))

    # 2. Tìm các khối dữ liệu có ngoặc vuông [...] hợp lệ
    # Chúng ta sẽ tách file theo các tiêu đề SAMPLE để kiểm tra từng khối
    blocks = re.split(r'(?:SAMPLE|--- SAMPLE)\s+\d+', content, flags=re.IGNORECASE)
    
    valid_data_ids = set()
    # Regex tìm ID và nội dung đi kèm để xác định ID nào thực sự có [ list ]
    # Cách này chính xác hơn nếu file của bạn có tiêu đề
    sample_blocks = re.findall(r'(?:SAMPLE|--- SAMPLE)\s+(\d+).*?(\[.*?\])', content, re.DOTALL | re.IGNORECASE)
    
    for sample_id, list_str in sample_blocks:
        # Kiểm tra xem list có đủ 20 phần tử không
        try:
            # Loại bỏ các khoảng trắng và kiểm tra số lượng dấu phẩy
            # Một list 20 phần tử sẽ có ít nhất 19 dấu phẩy
            if list_str.count(',') >= 19:
                valid_data_ids.add(int(sample_id))
        except:
            continue

    # 3. So sánh và tìm mẫu thiếu
    missing_ids = []
    error_format_ids = []

    for i in range(1, total_expected + 1):
        if i not in found_ids:
            missing_ids.append(i)
        elif i not in valid_data_ids:
            error_format_ids.append(i)

    # 4. Xuất báo cáo
    print(f"\nKẾT QUẢ KIỂM TRA:")
    print(f"- Tổng số mẫu dự kiến: {total_expected}")
    print(f"- Số mẫu tìm thấy tiêu đề: {len(found_ids)}")
    print(f"- Số mẫu có list [...] hợp lệ: {len(valid_data_ids)}")
    
    if missing_ids:
        print(f"\n[!] CÁC ID BỊ THIẾU HOÀN TOÀN (Không tìm thấy tiêu đề):")
        print(missing_ids)
    else:
        print(f"\n[OK] Không bị thiếu tiêu đề SAMPLE nào.")

    if error_format_ids:
        print(f"\n[!] CÁC ID CÓ TIÊU ĐỀ NHƯNG LỖI ĐỊNH DẠNG (List không đủ 20 số hoặc không có list):")
        print(error_format_ids)
    else:
        print(f"\n[OK] Tất cả các tiêu đề tìm thấy đều có dữ liệu hợp lệ.")

    if not missing_ids and not error_format_ids:
        print("\n=> File hoàn toàn khớp với Ground Truth!")
    else:
        print(f"\n=> TỔNG CỘNG CẦN XỬ LÝ LẠI {len(missing_ids) + len(error_format_ids)} MẪU.")

# Chạy kiểm tra với file của bạn
find_missing_samples('games_output.txt', total_expected=932)