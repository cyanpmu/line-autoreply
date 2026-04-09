"""
파우더브로우 이미지 분석 엔진 v4
- 선생님 기준 데이터 기반 비교
- OpenCV 실측 (Claude Vision 없음, 무료)
- 6가지 학생 공통 문제 자동 감지
"""

import cv2
import numpy as np
import json
import os

# ── 선생님 기준 데이터 로드 ──
REF_PATH = os.path.join(os.path.dirname(__file__), "teacher_reference.json")
with open(REF_PATH, "r", encoding="utf-8") as f:
    TEACHER_REF = json.load(f)

ZONE_JA = ["眉頭(前)", "眉頭〜眉上", "眉上(中央)", "眉上〜眉尾", "眉尾(尻)"]
ZONE_KO = ["미두(앞)", "미두~미상", "미상(중앙)", "미상~미미", "미미(꼬리)"]

# 선생님 실측 기준값
TEACHER_PIXEL_DARK = 61.3
TEACHER_PROFILE = [1.2, 27.8, 40.5, 35.0, 13.6]
TEACHER_GRADIENT_RANGE = 39.3
TEACHER_PIXEL_COUNT = 104613


def analyze_student_brow(image_bytes):
    """
    학생 사진 분석 → 선생님 기준과 비교 → 피드백 생성

    Args:
        image_bytes: 이미지 바이너리 (LINE에서 받은 것)

    Returns:
        dict: score, message_ja, message_ko, details
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "画像を読み取れません。もう一度送ってください。"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ── 배경 캘리브레이션 (모서리 4곳) ──
    corner_size_y = max(1, int(h * 0.08))
    corner_size_x = max(1, int(w * 0.08))
    corners = [
        gray[0:corner_size_y, 0:corner_size_x],
        gray[0:corner_size_y, w - corner_size_x:],
        gray[h - corner_size_y:, 0:corner_size_x],
        gray[h - corner_size_y:, w - corner_size_x:],
    ]
    bg_val = float(np.mean([np.mean(c) for c in corners]))

    # ── 눈썹 영역 감지 ──
    brow_mask = gray < (bg_val - 12)
    col_sums = np.sum(brow_mask, axis=0)
    brow_cols = np.where(col_sums > h * 0.03)[0]

    if len(brow_cols) < 20:
        return {"error": "眉毛が検出できませんでした。眉毛がはっきり写った写真を送ってください。"}

    bx1, bx2 = brow_cols[0], brow_cols[-1]
    row_sums = np.sum(brow_mask[:, bx1:bx2], axis=1)
    brow_rows = np.where(row_sums > (bx2 - bx1) * 0.03)[0]

    if len(brow_rows) < 5:
        return {"error": "眉毛の範囲が検出できませんでした。"}

    by1, by2 = brow_rows[0], brow_rows[-1]
    brow = gray[by1:by2, bx1:bx2]
    bh, bw = brow.shape

    # ── 5구역 분석 ──
    zone_w = bw // 5
    zones = []

    for i in range(5):
        zx = i * zone_w
        zw = zone_w if i < 4 else (bw - 4 * zone_w)
        zone = brow[:, zx:zx + zw]
        zh = zone.shape[0]

        dark_mean = max(0, bg_val - float(np.mean(zone)))
        z_sorted = np.sort(zone.flatten())
        dark_peak = max(0, bg_val - float(np.mean(z_sorted[:max(1, int(len(z_sorted) * 0.2))])))

        filled = np.sum(zone < (bg_val - 15)) / zone.size * 100
        empty = np.sum(zone > (bg_val - 5)) / zone.size * 100

        upper = zone[:zh // 2, :]
        lower = zone[zh // 2:, :]
        upper_dark = max(0, bg_val - float(np.mean(upper)))
        lower_dark = max(0, bg_val - float(np.mean(lower)))

        zones.append({
            "zone": i + 1,
            "dark": round(dark_mean, 1),
            "peak": round(dark_peak, 1),
            "filled_pct": round(filled, 1),
            "empty_pct": round(empty, 1),
            "upper_dark": round(upper_dark, 1),
            "lower_dark": round(lower_dark, 1),
        })

    # ── 전체 메트릭 ──
    brow_pixels = gray[brow_mask]
    student_pixel_dark = max(0, bg_val - float(np.mean(brow_pixels))) if len(brow_pixels) > 0 else 0
    student_pixel_count = int(np.sum(brow_mask))
    student_profile = [z["dark"] for z in zones]
    student_gradient_range = max(student_profile) - min(student_profile)
    student_peak_zone = student_profile.index(max(student_profile)) + 1

    # 선생님=80 기준 환산
    if TEACHER_PIXEL_DARK > 0:
        student_80scale = round((student_pixel_dark / TEACHER_PIXEL_DARK) * 80, 0)
    else:
        student_80scale = 0

    # ── 6가지 문제 감지 ──
    problems = []
    score = 100

    # 1. 레이어링 부족
    if student_80scale < 55:
        problems.append("layering")
        score -= 20
    elif student_80scale < 65:
        problems.append("layering_mild")
        score -= 10

    # 2. 眉頭 뭉침
    if zones[0]["dark"] > 8:
        problems.append("midu_dark")
        score -= 15
    elif zones[0]["dark"] > 5:
        problems.append("midu_dark_mild")
        score -= 8

    # 3. 윗단-아랫단 분리 (오렌지존에서 체크)
    oz = zones[2]  # 오렌지존
    if oz["upper_dark"] > 0 and oz["lower_dark"] > 0:
        vert_ratio = oz["upper_dark"] / oz["lower_dark"]
        if vert_ratio > 0.9 and vert_ratio < 1.1:
            # 윗단과 아랫단이 거의 같으면 수직 그라데이션 없음
            problems.append("no_vertical")
            score -= 10

    # 4. 구역 간 끊김
    for i in range(4):
        student_gap = abs(student_profile[i] - student_profile[i + 1])
        teacher_gap = abs(TEACHER_PROFILE[i] - TEACHER_PROFILE[i + 1])
        if student_gap > teacher_gap * 2 and student_gap > 10:
            problems.append(f"break_{i+1}_{i+2}")
            score -= 10

    # 5. 프로파일 평탄
    if student_gradient_range < 20:
        problems.append("flat_profile")
        score -= 15
    elif student_gradient_range < 30:
        problems.append("flat_profile_mild")
        score -= 8

    # 6. 피크 위치 불일치
    if student_peak_zone not in [2, 3, 4]:  # 피크가 앞이나 꼬리에 있으면
        problems.append("wrong_peak")
        score -= 10

    score = max(0, score)

    # ── 피드백 메시지 생성 ──
    msg_ja = _build_message_ja(zones, student_80scale, student_profile,
                                student_gradient_range, student_peak_zone,
                                student_pixel_count, problems, score)
    msg_ko = _build_message_ko(zones, student_80scale, student_profile,
                                student_gradient_range, student_peak_zone,
                                student_pixel_count, problems, score)

    return {
        "score": score,
        "message_ja": msg_ja,
        "message_ko": msg_ko,
        "student_80scale": student_80scale,
        "student_profile": student_profile,
        "student_gradient_range": round(student_gradient_range, 1),
        "student_peak_zone": student_peak_zone,
        "problems": problems,
        "zones": zones,
    }


def _build_message_ja(zones, scale80, profile, grad_range, peak_zone,
                       pixel_count, problems, score):
    """일본어 LINE 피드백 메시지"""
    refs = TEACHER_REF["student_common_problems"]

    msg = f"ご提出ありがとうございます！🙇\n添削させていただきます。\n\n"
    msg += f"📊 分析結果\n"
    msg += f"総合スコア: {score}/100\n"
    msg += f"濃さ: 先生=80 → 学生={scale80:.0f}\n"
    msg += f"グラデーション幅: 先生=40 → 学生={grad_range:.0f}\n"
    msg += f"一番濃いゾーン: {ZONE_JA[peak_zone - 1]}\n\n"

    # 구역별 수치
    msg += "【ゾーン別】\n"
    for i in range(5):
        z = zones[i]
        t = TEACHER_PROFILE[i]
        diff = z["dark"] - t
        icon = "✅" if abs(diff) < 5 else "🟡" if abs(diff) < 10 else "🔴"
        msg += f"{icon} {ZONE_JA[i]}: {z['dark']:.0f} (先生{t:.0f}, 差{diff:+.0f})\n"

    # 프로파일 비교
    msg += f"\n📈 プロファイル:\n"
    msg += f"学生: {' → '.join([f'{v:.0f}' for v in profile])}\n"
    msg += f"先生: {' → '.join([f'{v:.0f}' for v in TEACHER_PROFILE])}\n"

    # 문제별 피드백
    if problems:
        msg += "\n【改善ポイント】\n"
        for p in problems:
            if p == "layering":
                msg += f"\n🔴 {refs['1_레이어링부족']['feedback_ja']}\n"
            elif p == "layering_mild":
                msg += f"\n🟡 レイヤリングをあと2〜3回追加すると、もっと良くなります。\n"
            elif p == "midu_dark":
                msg += f"\n🔴 {refs['2_미두뭉침']['feedback_ja']}\n"
            elif p == "midu_dark_mild":
                msg += f"\n🟡 眉頭がやや濃いです。もう少し力を抜いてタッチしてください。\n"
            elif p == "no_vertical":
                msg += f"\n🔴 {refs['6_수직그라데이션없음']['feedback_ja']}\n"
            elif p.startswith("break_"):
                parts = p.split("_")
                z1, z2 = int(parts[1]) - 1, int(parts[2]) - 1
                msg += f"\n🔴 {ZONE_JA[z1]}→{ZONE_JA[z2]}間: {refs['4_구역간끊김']['feedback_ja']}\n"
            elif p == "flat_profile":
                msg += f"\n🔴 {refs['5_프로파일평탄']['feedback_ja']}\n"
            elif p == "flat_profile_mild":
                msg += f"\n🟡 明暗の差をもう少し大きくするとグラデーションがきれいになります。\n"
            elif p == "wrong_peak":
                msg += f"\n🟡 一番濃いゾーンが{ZONE_JA[peak_zone-1]}にあります。眉上（中央）が最も濃くなるようにしてください。\n"
    else:
        msg += "\n✨ 大きな問題は見つかりませんでした！このまま練習を続けてください。\n"

    msg += "\n引き続き練習を頑張ってください！\n何かご不明な点がございましたら、お気軽にご質問ください☺️"
    return msg


def _build_message_ko(zones, scale80, profile, grad_range, peak_zone,
                       pixel_count, problems, score):
    """한국어 메시지 (선생님 검토용)"""
    msg = f"📊 분석 결과 (점수: {score}/100)\n"
    msg += f"진하기: 선생님=80 → 학생={scale80:.0f}\n"
    msg += f"그라데이션 폭: 선생님=40 → 학생={grad_range:.0f}\n"
    msg += f"피크: {ZONE_KO[peak_zone - 1]}\n\n"

    for i in range(5):
        z = zones[i]
        t = TEACHER_PROFILE[i]
        diff = z["dark"] - t
        icon = "✅" if abs(diff) < 5 else "🟡" if abs(diff) < 10 else "🔴"
        msg += f"{icon} {ZONE_KO[i]}: {z['dark']:.0f} (선생님{t:.0f}, {diff:+.0f})\n"

    msg += f"\n학생: {' → '.join([f'{v:.0f}' for v in profile])}\n"
    msg += f"선생님: {' → '.join([f'{v:.0f}' for v in TEACHER_PROFILE])}\n"

    if problems:
        msg += f"\n문제: {', '.join(problems)}"

    return msg
