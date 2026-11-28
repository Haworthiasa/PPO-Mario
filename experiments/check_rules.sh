#!/usr/bin/env bash
# Kiểm các quy định của đề (de-bai.md, Mục 3.5) và quy ước của nhóm trước khi chạy một lượt mới, và sau đó cho các thư
# mục lượt đã chạy. Thoát mã 1 nếu có điều sai.
#   bash experiments/check_rules.sh [thư mục lượt ...]
cd "$(dirname "$0")/.."
START=4093e7d   # commit mã khởi đầu của thầy
# sha256 của `git diff 4093e7d -- mario_ppo/env.py` ở bản đã duyệt: chỉ thêm bốn mẹo của nhóm (cắt lượt khi không tiến,
# thưởng cờ, không cộng x khi rơi rồi chết, bắt đầu lại ngay trước chỗ vừa chết) trong khối `if not eval_mode:` và dòng
# import của chúng. Mọi thay đổi khác của env.py làm số này đổi. Các bản trước: ba mẹo
# 5e31e6d383f0af34b85823c76329f853378099073249aeadd60e685aec97b401, hai mẹo
# 1a64c8261ef31b837f331fac98587c9a67ecdf8ed875911e362d22805764b8c2.
ENV_DIFF_OK=4408ebffa35257a38fa4e9fdf21d31a2f2f7a5dfcfe513857112aa4ff988fbe4
# sha256 của `git diff HEAD -- mario_ppo/wrappers.py` khi wrappers.py chỉ thêm PitFallClaw và FailureRestart (28/11/2026,
# chưa commit): PitFallClaw chỉ sửa phần thưởng; FailureRestart chỉ chọn trạng thái bắt đầu lượt (đọc RAM để chọn lúc
# lưu), quan sát vẫn là khung hình của trình giả lập. Sau khi commit thì diff rỗng và phép kiểm "không đổi" dưới đây
# tự đúng; số này chỉ dùng cho khoảng thời gian trước khi commit. Bản chỉ có PitFallClaw:
# 117e430e88c25260233f86c88dfc0db9e259e5eb48a8d45cc5b94173f9ecb62b.
WRAPPERS_DIFF_OK=db30e7076492a3e2b1988fbda50edb93f36b91c2fac21ebd38a18e410e782137
fail=0
ok() { echo "  đúng: $1"; }
bad() { echo "  SAI: $1"; fail=1; }

echo "Quy định 3.5.1: không sửa evaluate.py và phần eval_mode của env.py; mẹo đặt trong if not eval_mode"
if git diff --quiet "$START" -- mario_ppo/evaluate.py; then ok "evaluate.py giống mã khởi đầu"; else bad "evaluate.py đã bị sửa"; fi
h=$(git diff "$START" -- mario_ppo/env.py | sha256sum | cut -d' ' -f1)
if [ "$h" = "$ENV_DIFF_OK" ]; then ok "env.py chỉ khác mã khởi đầu ở bốn mẹo đã duyệt trong if not eval_mode"
else bad "env.py có thay đổi mới so với bản đã duyệt (git diff $START -- mario_ppo/env.py)"; fi

echo "Quy định 3.5.2: đầu vào của mạng chỉ là khung hình đã tiền xử lý"
if git diff --quiet HEAD -- mario_ppo/model.py; then ok "model.py không đổi so với HEAD"
else bad "model.py có thay đổi chưa commit; kiểm lại đầu vào của mạng"; fi
hw=$(git diff HEAD -- mario_ppo/wrappers.py | sha256sum | cut -d' ' -f1)
if git diff --quiet HEAD -- mario_ppo/wrappers.py; then ok "wrappers.py không đổi so với HEAD"
elif [ "$hw" = "$WRAPPERS_DIFF_OK" ]; then ok "wrappers.py chỉ thêm PitFallClaw và FailureRestart đã duyệt (không đổi quan sát)"
else bad "wrappers.py có thay đổi chưa duyệt; kiểm lại đầu vào của mạng"; fi
if grep -nqE '\bram\b|info\[' mario_ppo/model.py; then bad "model.py nhắc tới RAM hoặc info"; else ok "model.py không đọc RAM hay info"; fi

echo "Quy định 3.5.3: không dùng checkpoint bên ngoài (mọi lượt học từ đầu)"
for f in configs/group-*-tuning.yaml configs/group-*-fast.yaml; do
  if grep -qE '^resume: null' "$f"; then ok "$f: resume null"; else bad "$f: resume khác null"; fi
done
for d in "$@"; do
  [ -f "$d/config.yaml" ] || continue
  if grep -qE '^resume: null' "$d/config.yaml"; then ok "$d: resume null"; else bad "$d: resume khác null"; fi
done

echo "Quy định 3.5.4: checkpoint tái tạo được từ cấu hình"
for d in "$@"; do
  [ -d "$d" ] || continue
  if [ -f "$d/config.yaml" ] && [ -f "$d/code-version.txt" ]; then ok "$d có config.yaml và code-version.txt"
  else bad "$d thiếu config.yaml hoặc code-version.txt"; fi
done
if ! git diff --quiet HEAD -- mario_ppo/train_joint.py; then
  echo "  lưu ý: train_joint.py có thay đổi chưa commit (lr_control, cap_final, failure_restart); phải commit trước khi nộp mạng nào từ các lượt này"
fi

echo "Quy ước của nhóm: train.py giữ nguyên; submission/ chỉ đổi bằng commit có chủ ý (lần gần nhất: 29/11/2026, 1-1 và 4-1)"
if [ -z "$(git status --porcelain submission)" ] && git diff --quiet HEAD -- submission; then ok "submission/ không có thay đổi chưa commit"
else bad "submission/ có thay đổi chưa commit"; fi
if git diff --quiet HEAD -- mario_ppo/train.py; then ok "train.py không đổi"; else bad "train.py đã đổi"; fi

[ $fail -eq 0 ] && echo "KẾT QUẢ: không có điều sai" || echo "KẾT QUẢ: CÓ điều sai, không chạy lượt mới"
exit $fail
