"""
파우더브로우 이미지 분석 엔진 v5
- 선생님 기준 데이터 기반 비교
- 복수 눈썹 자동 감지 (3패턴 과제)
- 패턴 구분 안 되면 솔직히 말함
- OpenCV 실측 (무료)
"""

import cv2
import numpy as np
import json
import os

REF_PATH = os.path.join(os.path.dirname(__file__), "teacher_reference.json")
with open(REF_PATH, "r", encoding="utf-8") as f:
    TEACHER_REF = json.load(f)

ZONE_JA = ["眉頭(前)", "眉頭〜眉上", "眉上(中央)", "眉上〜眉尾", "眉尾(尻)"]
ZONE_KO = ["미두(앞)", "미두~미상", "미상(중앙)", "미상~미미", "미미(꼬리)"]

TEACHER_PIXEL_DARK = 61.3
TEACHER_PROFILE = [1.2, 27.8, 40.5, 35.0, 13.6]
TEACHER_GRADIENT_RANGE = 39.3


def _analyze_one_brow(gray, bg_val):
    """단일 눈썹 영역 분석"""
    h, w = gray.shape

    brow_mask = gray < (bg_val - 12)
    col_sums = np.sum(brow_mask, axis=0)
    brow_cols = np.where(col_sums > h * 0.03)[0]
    if len(brow_cols) < 20:
        return None

    bx1, bx2 = brow_cols[0], brow_cols[-1]
    row_sums = np.sum(brow_mask[:, bx1:bx2], axis=1)
    brow_rows = np.where(row_sums > (bx2 - bx1) * 0.03)[0]
    if len(brow_rows) < 5:
        return None

    by1, by2 = brow_rows[0], brow_rows[-1]
    brow = gray[by1:by2, bx1:bx2]
    bh, bw = brow.shape
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
            "zone": i + 1, "dark": round(dark_mean, 1), "peak": round(dark_peak, 1),
            "filled_pct": round(filled, 1), "empty_pct": round(empty, 1),
            "upper_dark": round(upper_dark, 1), "lower_dark": round(lower_dark, 1),
        })

    brow_pixels = gray[brow_mask]
    pixel_dark = max(0, bg_val - float(np.mean(brow_pixels))) if len(brow_pixels) > 0 else 0
    profile = [z["dark"] for z in zones]
    grad_range = max(profile) - min(profile)
    peak_zone = profile.index(max(profile)) + 1

    scale80 = round((pixel_dark / TEACHER_PIXEL_DARK) * 80, 0) if TEACHER_PIXEL_DARK > 0 else 0

    return {
        "zones": zones, "scale80": scale80, "profile": profile,
        "grad_range": round(grad_range, 1), "peak_zone": peak_zone,
        "pixel_dark": round(pixel_dark, 1),
    }


def _detect_problems(data):
    """문제 감지"""
    problems = []
    score = 100
    zones = data["zones"]
    profile = data["profile"]

    # 1. 레이어링 부족
    if data["scale80"] < 55:
        problems.append("layering")
        score -= 20
    elif data["scale80"] < 65:
        problems.append("layering_mild")
        score -= 10

    # 2. 眉頭 뭉침
    if zones[0]["dark"] > 8:
        problems.append("midu_dark")
        score -= 15
    elif zones[0]["dark"] > 5:
        problems.append("midu_dark_mild")
        score -= 8

    # 3. 수직 그라데이션 없음
    oz = zones[2]
    if oz["lower_dark"] > 0:
        vert_ratio = oz["upper_dark"] / oz["lower_dark"] if oz["lower_dark"] > 0 else 1
        if 0.85 < vert_ratio < 1.15:
            problems.append("no_vertical")
            score -= 10

    # 4. 구역 간 끊김
    for i in range(4):
        s_gap = abs(profile[i] - profile[i + 1])
        t_gap = abs(TEACHER_PROFILE[i] - TEACHER_PROFILE[i + 1])
        if s_gap > max(t_gap * 2, 10):
            problems.append(f"break_{i+1}_{i+2}")
            score -= 10

    # 5. 프로파일 평탄
    if data["grad_range"] < 15:
        problems.append("flat_profile")
        score -= 15
    elif data["grad_range"] < 25:
        problems.append("flat_profile_mild")
        score -= 8

    # 6. 피크 위치
    if data["peak_zone"] not in [2, 3, 4]:
        problems.append("wrong_peak")
        score -= 10

    return problems, max(0, score)


def _check_pattern_distinction(brow_results):
    """
    3개 눈썹의 패턴이 구분되는지 체크
    구분 안 되면 솔직히 말함
    """
    if len(brow_results) < 2:
        return None

    profiles = [b["profile"] for b in brow_results]
    scales = [b["scale80"] for b in brow_results]

    # 진하기 순 정렬
    sorted_idx = sorted(range(len(scales)), key=lambda i: scales[i], reverse=True)

    # 패턴 간 유사도: 프로파일 차이
    distinctions = []
    for i in range(len(brow_results)):
        for j in range(i + 1, len(brow_results)):
            diff = sum(abs(profiles[i][k] - profiles[j][k]) for k in range(5))
            distinctions.append((i, j, diff))

    # 구분 판정
    indistinguishable_pairs = [(i, j) for i, j, d in distinctions if d < 15]

    result = {
        "sorted_by_darkness": sorted_idx,
        "distinctions": distinctions,
        "indistinguishable": indistinguishable_pairs,
        "can_distinguish_all": len(indistinguishable_pairs) == 0,
    }

    # 패턴 추정
    if len(brow_results) == 3:
        darkest = sorted_idx[0]
        middle = sorted_idx[1]
        lightest = sorted_idx[2]

        # 가장 진한 게 SOFT, 가장 연한 게 NATURAL로 추정
        # 단, 차이가 충분해야 함
        dark_diff = scales[darkest] - scales[lightest]
        mid_diff1 = abs(scales[darkest] - scales[middle])
        mid_diff2 = abs(scales[middle] - scales[lightest])

        if dark_diff > 10:
            result["guess"] = {
                darkest: "SOFT(推定)",
                middle: "MIX(推定)",
                lightest: "NATURAL(推定)",
            }
            if mid_diff1 < 8 or mid_diff2 < 8:
                result["uncertain_pairs"] = True
        else:
            result["guess"] = None
            result["too_similar"] = True

    return result


def analyze_student_brow(image_bytes):
    """메인 분석 함수"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "画像を読み取れません。もう一度送ってください。"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 배경 캘리브레이션
    cs_y = max(1, int(h * 0.06))
    cs_x = max(1, int(w * 0.06))
    corners = [gray[0:cs_y, 0:cs_x], gray[0:cs_y, w-cs_x:],
               gray[h-cs_y:, 0:cs_x], gray[h-cs_y:, w-cs_x:]]
    bg_val = float(np.mean([np.mean(c) for c in corners]))

    # 복수 눈썹 감지: 세로로 나뉘어 있는지 체크
    brow_mask = gray < (bg_val - 12)
    row_density = np.sum(brow_mask, axis=1) / w

    # 빈 행 찾기 (눈썹 사이 간격)
    # 라텍스 텍스처 때문에 완전히 0이 안 되므로 높은 기준 사용
    # 밀도 중간값 기준으로 이진화
    median_density = float(np.median(row_density[row_density > 0.01]))
    brow_threshold = max(0.10, median_density * 0.4)
    in_brow = row_density > brow_threshold
    transitions = np.diff(in_brow.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    if len(starts) == 0 and in_brow[0]:
        starts = np.array([0])
    if len(ends) == 0 and in_brow[-1]:
        ends = np.array([h - 1])
    if len(starts) > 0 and len(ends) > 0:
        if starts[0] > ends[0]:
            starts = np.concatenate([[0], starts])
        if ends[-1] < starts[-1]:
            ends = np.concatenate([ends, [h - 1]])

    min_len = min(len(starts), len(ends))
    brow_regions = []
    for i in range(min_len):
        region_h = ends[i] - starts[i]
        if region_h > h * 0.05:  # 최소 높이 (3눈썹이면 각각 ~30%)
            brow_regions.append((starts[i], ends[i]))

    # ── 단일 눈썹 ──
    if len(brow_regions) <= 1:
        data = _analyze_one_brow(gray, bg_val)
        if data is None:
            return {"error": "眉毛が検出できませんでした。眉毛がはっきり写った写真を送ってください。"}

        problems, score = _detect_problems(data)
        msg_ja = _build_single_message_ja(data, problems, score)
        return {
            "type": "single",
            "score": score, "message_ja": msg_ja,
            "problems": problems, **data,
        }

    # ── 복수 눈썹 (3패턴 과제) ──
    brow_results = []
    for idx, (y1, y2) in enumerate(brow_regions[:3]):
        pad = int((y2 - y1) * 0.05)
        region = gray[max(0, y1 - pad):min(h, y2 + pad), :]
        data = _analyze_one_brow(region, bg_val)
        if data:
            problems, score = _detect_problems(data)
            data["problems"] = problems
            data["score"] = score
            data["label"] = f"{idx+1}番目"
            brow_results.append(data)

    if len(brow_results) == 0:
        return {"error": "眉毛が検出できませんでした。"}

    # 패턴 구분 체크
    pattern_check = _check_pattern_distinction(brow_results) if len(brow_results) >= 2 else None

    msg_ja = _build_multi_message_ja(brow_results, pattern_check)
    avg_score = round(sum(b["score"] for b in brow_results) / len(brow_results))

    return {
        "type": "multi",
        "count": len(brow_results),
        "score": avg_score,
        "message_ja": msg_ja,
        "brows": brow_results,
        "pattern_check": pattern_check,
    }


def _build_single_message_ja(data, problems, score):
    """단일 눈썹 일본어 피드백"""
    refs = TEACHER_REF["student_common_problems"]
    profile = data["profile"]

    msg = f"ご提出ありがとうございます！🙇\n添削させていただきます。\n\n"
    msg += f"📊 分析結果\n"
    msg += f"総合スコア: {score}/100\n"
    msg += f"濃さ: 先生=80 → 学生={data['scale80']:.0f}\n"
    msg += f"グラデーション幅: 先生=40 → 学生={data['grad_range']:.0f}\n\n"

    msg += "【ゾーン別】\n"
    for i in range(5):
        z = data["zones"][i]
        t = TEACHER_PROFILE[i]
        diff = z["dark"] - t
        icon = "✅" if abs(diff) < 5 else "🟡" if abs(diff) < 10 else "🔴"
        msg += f"{icon} {ZONE_JA[i]}: {z['dark']:.0f} (先生{t:.0f}, 差{diff:+.0f})\n"

    msg += f"\n📈 学生: {' → '.join([f'{v:.0f}' for v in profile])}\n"
    msg += f"📈 先生: {' → '.join([f'{v:.0f}' for v in TEACHER_PROFILE])}\n"

    msg += _problems_to_ja(problems, refs)

    msg += "\n引き続き練習を頑張ってください！\nご不明な点はお気軽にどうぞ☺️"
    return msg


def _build_multi_message_ja(brow_results, pattern_check):
    """복수 눈썹 일본어 피드백"""
    refs = TEACHER_REF["student_common_problems"]
    n = len(brow_results)

    msg = f"ご提出ありがとうございます！🙇\n{n}つの眉を分析しました。\n\n"

    # ── 전체 요약 ──
    avg_score = round(sum(b["score"] for b in brow_results) / n)
    msg += f"📊 総合スコア: {avg_score}/100\n\n"

    # ── 패턴 구분 ──
    if pattern_check:
        if pattern_check.get("too_similar"):
            msg += "⚠️ 3つの眉の違いがまだはっきりしていません。\n"
            msg += "レイヤリングが足りないと、パターンの特徴が現れません。\n"
            msg += "まずしっかり重ねてから、点の大きさと間隔で\n"
            msg += "3パターンの違いを出しましょう。\n\n"
        elif pattern_check.get("guess"):
            guess = pattern_check["guess"]
            msg += "【パターン推定】\n"
            for idx, label in guess.items():
                b = brow_results[idx]
                msg += f"  {b['label']}: {label} (濃さ{b['scale80']:.0f})\n"
            if pattern_check.get("uncertain_pairs"):
                msg += "  ※ 一部のパターンの区別がまだ弱いです\n"
            msg += "\n"
        elif not pattern_check.get("can_distinguish_all"):
            msg += "⚠️ 一部のパターンが似すぎて区別が難しいです。\n"
            msg += "ソフトは点を小さく密に、ナチュラルは点を大きく間隔広く、\n"
            msg += "ミックスはその中間を意識してください。\n\n"

    # ── 공통 문제 ──
    all_problems = set()
    for b in brow_results:
        all_problems.update(b["problems"])

    common = []
    for p in ["layering", "layering_mild", "flat_profile", "flat_profile_mild"]:
        if all(p in b["problems"] for b in brow_results):
            common.append(p)

    if common:
        msg += "【共通の課題】\n"
        if "layering" in common:
            scales = ", ".join([f"{b['label']}={b['scale80']:.0f}" for b in brow_results])
            msg += f"🔴 全体的にレイヤリングが足りません\n"
            msg += f"  先生=80 → {scales}\n"
            msg += f"  同じ弱い力でもう3回以上重ねてください。\n"
            msg += f"  重ねれば重ねるほど点が面になります。\n\n"
        elif "layering_mild" in common:
            msg += f"🟡 もう少しレイヤリングを追加するとさらに良くなります。\n\n"

        if "flat_profile" in common:
            msg += f"🔴 グラデーションの幅が小さいです。\n"
            msg += f"  眉頭をもっと薄く、眉上をもっと濃くして\n"
            msg += f"  明暗の差を大きくしてください。\n\n"
        elif "flat_profile_mild" in common:
            msg += f"🟡 明暗の差をもう少し大きくするときれいになります。\n\n"

    # ── 개별 분석 ──
    msg += "【個別分析】\n"
    for b in brow_results:
        profile = b["profile"]
        msg += f"\n── {b['label']} (スコア{b['score']}, 濃さ{b['scale80']:.0f}) ──\n"
        msg += f"  {' → '.join([f'{v:.0f}' for v in profile])}\n"

        # 공통 문제 제외한 개별 문제만
        individual = [p for p in b["problems"] if p not in common]
        if individual:
            for p in individual:
                if p == "midu_dark":
                    msg += f"  🔴 眉頭が濃すぎます（{b['zones'][0]['dark']:.0f}, 先生は1）\n"
                    msg += f"    → 力を抜いて、肌からスーッとつながるように\n"
                elif p == "midu_dark_mild":
                    msg += f"  🟡 眉頭がやや濃いです\n"
                elif p == "no_vertical":
                    msg += f"  🔴 上段と下段の明暗差がありません\n"
                    msg += f"    → 下段を先にしっかり、上はふんわり\n"
                elif p.startswith("break_"):
                    parts = p.split("_")
                    z1, z2 = int(parts[1]) - 1, int(parts[2]) - 1
                    msg += f"  🔴 {ZONE_JA[z1]}→{ZONE_JA[z2]}でグラデーション途切れ\n"
                    msg += f"    → 2〜3mmのブレンディング区間でつなげて\n"
                elif p == "wrong_peak":
                    msg += f"  🟡 一番濃い部分が{ZONE_JA[b['peak_zone']-1]}にあります\n"
                    msg += f"    → 眉上(中央)が最も濃くなるように\n"
        else:
            if not common:
                msg += f"  ✅ 大きな問題はありません！\n"

    msg += f"\n先生のプロファイル参考:\n"
    msg += f"📈 {' → '.join([f'{v:.0f}' for v in TEACHER_PROFILE])}\n"
    msg += f"\n引き続き練習を頑張ってください！\nご不明な点はお気軽にどうぞ☺️"
    return msg


def _problems_to_ja(problems, refs):
    """문제 리스트 → 일본어 피드백"""
    if not problems:
        return "\n✨ 大きな問題は見つかりませんでした！\n"

    msg = "\n【改善ポイント】\n"
    for p in problems:
        if p == "layering":
            msg += f"\n🔴 {refs['1_레이어링부족']['feedback_ja']}\n"
        elif p == "layering_mild":
            msg += f"\n🟡 レイヤリングをあと2〜3回追加するともっと良くなります。\n"
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
            msg += f"\n🟡 一番濃いゾーンが眉上(中央)以外にあります。眉上が最も濃くなるようにしてください。\n"

    return msg
