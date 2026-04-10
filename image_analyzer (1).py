"""
파우더브로우 이미지 분석기 v2
- 선생님 피드백 반영 강화 채점
- 오렌지존 분리, 점 간격 과밀, 압/깊이 과다, 얼룩 감지 추가
- OpenCV만 사용 (무료)
"""

import cv2
import numpy as np
import json
import os

# 선생님 기준 데이터
TEACHER_REF = {
    "profile_80": [1, 28, 40, 35, 14],  # 80기준 Z1~Z5
    "raw_darkness": 61.3,
    "pixel_count": 104613,
    "gradient_range": 40,
}

PATTERN_REFS = {
    "SOFT": {
        "darkness": [121, 128, 136, 129, 116],
        "density":  [26.1, 53.0, 67.4, 67.5, 40.0],
        "description": "점 간격 좁고 작게, 매그넘 니들, 중앙~하부 명암 강조",
    },
    "NATURAL": {
        "darkness": [120, 122, 123, 121, 122],
        "density":  [25.1, 54.0, 57.4, 50.6, 35.2],
        "description": "점 크고 간격 넓음, 포인트 니들만, 피부 투과 느낌",
    },
    "MIX": {
        "darkness": [125, 127, 129, 128, 126],
        "density":  [32.1, 63.7, 62.8, 53.3, 39.9],
        "description": "SOFT+NATURAL 결합, 매그넘+포인트",
    },
}

# 6가지 학생 공통 문제
COMMON_PROBLEMS = {
    "layering": "레이어링 부족 — 진하기 80기준 55 미만",
    "midu_dark": "미두 뭉침 — 眉頭 진하기 8 초과 (선생님은 1~3)",
    "separation": "윗단-아랫단 분리 — 빈 띠, 두 줄 느낌",
    "gap": "구역 간 끊김 — 인접 구역 차이가 선생님의 2배",
    "flat_profile": "프로파일 평탄 — 그라데이션 폭 20 미만",
    "no_vertical": "수직 그라데이션 없음 — 윗단=아랫단",
}


def detect_brows(image):
    """눈썹 영역 감지 (복수 눈썹 지원)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    bg_value = np.median(gray)
    threshold = bg_value * 0.75
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    brows = []
    min_area = image.shape[0] * image.shape[1] * 0.005
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h
            if 1.5 < aspect < 8.0:
                brows.append({
                    "contour": cnt,
                    "bbox": (x, y, w, h),
                    "area": area,
                })
    
    brows.sort(key=lambda b: b["bbox"][0])
    return brows


def analyze_zones(gray, brow, num_zones=5):
    """5구역 분할 분석"""
    x, y, w, h = brow["bbox"]
    zone_width = w // num_zones
    
    bg_value = float(np.median(gray))
    
    zones = []
    for i in range(num_zones):
        zx = x + i * zone_width
        zw = zone_width if i < num_zones - 1 else (w - i * zone_width)
        
        roi = gray[y:y+h, zx:zx+zw]
        mask_roi = np.zeros_like(roi)
        shifted = brow["contour"].copy()
        shifted[:, 0, 0] -= zx
        shifted[:, 0, 1] -= y
        cv2.drawContours(mask_roi, [shifted], -1, 255, -1)
        
        brow_pixels = roi[mask_roi > 0]
        
        if len(brow_pixels) == 0:
            zones.append({"darkness": 0, "density": 0, "pixel_count": 0})
            continue
        
        darkness = float(bg_value - np.mean(brow_pixels))
        darkness = max(0, darkness)
        
        dark_threshold = bg_value * 0.85
        dark_count = np.sum(brow_pixels < dark_threshold)
        density = (dark_count / len(brow_pixels)) * 100 if len(brow_pixels) > 0 else 0
        
        zones.append({
            "darkness": round(darkness, 1),
            "density": round(density, 1),
            "pixel_count": int(len(brow_pixels)),
        })
    
    return zones


def normalize_to_80(zones):
    """선생님 80기준으로 정규화"""
    max_dark = max(z["darkness"] for z in zones) if zones else 1
    if max_dark == 0:
        return [0] * len(zones)
    scale = 80 / max_dark
    return [round(z["darkness"] * scale, 1) for z in zones]


def generate_feedback_v2(zones, pattern="SOFT"):
    """v2 강화 피드백 생성"""
    ref = PATTERN_REFS.get(pattern, PATTERN_REFS["SOFT"])
    ref_dark = ref["darkness"]
    ref_dens = ref["density"]
    
    feedback = []
    problems = []
    total_deduction = 0
    
    stu_dark = [z["darkness"] for z in zones]
    stu_dens = [z["density"] for z in zones]
    
    # 80기준 정규화
    profile_80 = normalize_to_80(zones)
    teacher_profile = TEACHER_REF["profile_80"]
    
    # === 1. 전체 레이어링 부족 ===
    avg_dark_80 = sum(profile_80) / 5
    if avg_dark_80 < 35:
        feedback.append({
            "type": "critical",
            "area": "전체 레이어링",
            "weight": 15,
            "msg": f"레이어링이 심하게 부족합니다 (80기준 평균 {avg_dark_80:.0f}, 선생님 기준 55+). 같은 약한 힘으로 5회 이상 쌓아주세요.",
        })
        problems.append("layering")
        total_deduction += 15
    elif avg_dark_80 < 55:
        feedback.append({
            "type": "critical",
            "area": "레이어링 부족",
            "weight": 12,
            "msg": f"레이어링이 부족합니다 (80기준 평균 {avg_dark_80:.0f}, 선생님 기준 55+). 패스를 추가해주세요.",
        })
        problems.append("layering")
        total_deduction += 12
    
    # === 2. 전체 과다 (압/깊이) ===
    stu_avg = sum(stu_dark) / 5
    ref_avg = sum(ref_dark) / 5
    diff_avg = stu_avg - ref_avg
    
    if diff_avg > 15:
        feedback.append({
            "type": "critical",
            "area": "압/깊이 과다",
            "weight": 12,
            "msg": f"전체적으로 너무 진합니다 (차이 +{diff_avg:.0f}). 힘을 빼고 아기 볼 만지듯 부드럽게 터치해주세요. 진한 부분은 세게 누르는 게 아니라, 약한 힘으로 여러 번 쌓는 것!",
        })
        total_deduction += 12
    elif diff_avg > 8:
        feedback.append({
            "type": "warning",
            "area": "약간 과다",
            "weight": 6,
            "msg": f"약간 진한 편입니다 (차이 +{diff_avg:.0f}). 압력을 조금 줄여주세요.",
        })
        total_deduction += 6
    
    # === 3. 오렌지존 분리 ===
    z3 = stu_dark[2]
    z2 = stu_dark[1]
    z4 = stu_dark[3]
    ref_jump_2 = ref_dark[2] - ref_dark[1]
    ref_jump_4 = ref_dark[2] - ref_dark[3]
    stu_jump_2 = z3 - z2
    stu_jump_4 = z3 - z4
    
    if stu_jump_2 > ref_jump_2 * 2 + 5 or stu_jump_4 > ref_jump_4 * 2 + 5:
        feedback.append({
            "type": "critical",
            "area": "오렌지존 분리",
            "weight": 15,
            "msg": "오렌지존(Z3)이 인접 구역과 분리되어 보입니다. 진한 부분이 따로 떠 있는 느낌. 오렌지존에서 위로/옆으로 레이어링을 이어서 쌓아 블렌딩 구간(2~3mm)을 만들어주세요.",
        })
        problems.append("separation")
        total_deduction += 15
    
    # === 4. 점 간격 과밀 ===
    stu_avg_dens = sum(stu_dens) / 5
    ref_avg_dens = sum(ref_dens) / 5
    dens_diff = stu_avg_dens - ref_avg_dens
    
    if dens_diff > 15:
        feedback.append({
            "type": "critical",
            "area": "점 간격 과밀",
            "weight": 12,
            "msg": f"점 간격이 너무 좁아서 그라데이션 느낌이 없습니다 (밀도 {stu_avg_dens:.0f}% vs 정답 {ref_avg_dens:.0f}%). 윗단은 점 간격을 넓혀서 피부가 보이도록 풀어주세요.",
        })
        total_deduction += 12
    elif dens_diff > 8:
        feedback.append({
            "type": "warning",
            "area": "밀도 높음",
            "weight": 6,
            "msg": f"전체 밀도가 높은 편 ({stu_avg_dens:.0f}% vs 정답 {ref_avg_dens:.0f}%). 점 간격을 약간 넓혀주세요.",
        })
        total_deduction += 6
    
    # === 5. 얼룩/불균일 ===
    stu_range = max(stu_dark) - min(stu_dark)
    ref_range = max(ref_dark) - min(ref_dark)
    
    if diff_avg > 10 and stu_range > ref_range * 1.5:
        feedback.append({
            "type": "critical",
            "area": "얼룩/불균일",
            "weight": 10,
            "msg": f"진하면서 구역 간 편차 커서 얼룩져 보입니다 (범위 {stu_range:.0f} vs 정답 {ref_range:.0f}). 힘을 균일하게, 천천히 정확하게 쌓아주세요.",
        })
        total_deduction += 10
    
    # === 6. 앞머리 그라데이션 ===
    stu_gradient = stu_dark[2] - stu_dark[0]
    ref_gradient = ref_dark[2] - ref_dark[0]
    
    if pattern in ("SOFT", "MIX"):
        if stu_gradient < 3:
            feedback.append({
                "type": "critical",
                "area": "앞머리 그라데이션 없음",
                "weight": 15,
                "msg": f"미두→오렌지존 그라데이션이 거의 없습니다 (차이 {stu_gradient:.0f}, 정답 {ref_gradient:.0f}). 앞머리는 극히 연하게, 중앙은 여러 번 쌓아 진하게!",
            })
            total_deduction += 15
        elif stu_gradient < ref_gradient * 0.4:
            feedback.append({
                "type": "critical",
                "area": "그라데이션 부족",
                "weight": 12,
                "msg": f"미두→오렌지존 그라데이션 부족 ({stu_gradient:.0f} vs 정답 {ref_gradient:.0f}). 미두 점 간격 넓히고, 중앙 레이어링 추가!",
            })
            total_deduction += 12
        elif stu_gradient < ref_gradient * 0.6:
            feedback.append({
                "type": "warning",
                "area": "그라데이션 약함",
                "weight": 8,
                "msg": f"그라데이션이 약합니다 ({stu_gradient:.0f} vs 정답 {ref_gradient:.0f}).",
            })
            total_deduction += 8
    
    # === 7. Z1→Z2→Z3 연속성 ===
    z1z2 = stu_dark[1] - stu_dark[0]
    z2z3 = stu_dark[2] - stu_dark[1]
    
    if z1z2 < 2 and z2z3 > 8:
        feedback.append({
            "type": "critical",
            "area": "앞머리~가운데 끊김",
            "weight": 12,
            "msg": "미두~미상이 거의 같은 진하기인데 오렌지존에서 갑자기 진해집니다. 앞머리→가운데 점진적 그라데이션 필요!",
        })
        problems.append("gap")
        total_deduction += 12
    
    # === 8. 미두 뭉침 ===
    if profile_80[0] > 8:
        feedback.append({
            "type": "warning",
            "area": "미두 뭉침",
            "weight": 8,
            "msg": f"앞머리가 너무 진합니다 (80기준 {profile_80[0]:.0f}, 선생님은 1~3). '사라지듯' 마감 필요. 점 3~5개만 극히 적게, 힘 빼고!",
        })
        problems.append("midu_dark")
        total_deduction += 8
    
    if stu_dens[0] > ref_dens[0] * 1.4:
        feedback.append({
            "type": "warning",
            "area": "미두 밀도",
            "weight": 6,
            "msg": f"앞머리 밀도 과다 ({stu_dens[0]:.0f}% vs 정답 {ref_dens[0]:.0f}%). 점 간격 넓혀서 스르르 나타나는 느낌으로!",
        })
        total_deduction += 6
    
    # === 9. 프로파일 평탄 ===
    grad_range = max(profile_80) - min(profile_80)
    if grad_range < 15:
        feedback.append({
            "type": "critical",
            "area": "프로파일 평탄",
            "weight": 10,
            "msg": f"그라데이션 폭이 너무 좁습니다 (폭 {grad_range:.0f}, 선생님 40). 오렌지존은 진하게 쌓고, 앞머리/꼬리는 힘 빼세요.",
        })
        problems.append("flat_profile")
        total_deduction += 10
    elif grad_range < 20:
        feedback.append({
            "type": "warning",
            "area": "프로파일 약간 평탄",
            "weight": 6,
            "msg": f"그라데이션 폭이 약간 좁습니다 (폭 {grad_range:.0f}, 선생님 40).",
        })
        problems.append("flat_profile")
        total_deduction += 6
    
    # === 10. 꼬리 페이드아웃 ===
    tail_drop = stu_dark[3] - stu_dark[4]
    ref_tail_drop = ref_dark[3] - ref_dark[4]
    
    if tail_drop < 0 and ref_tail_drop > 3:
        feedback.append({
            "type": "critical",
            "area": "꼬리 역전",
            "weight": 10,
            "msg": "꼬리가 중간보다 진합니다! 꼬리로 갈수록 점 줄이고 간격 넓혀서 페이드아웃!",
        })
        total_deduction += 10
    
    # === 11. 구역별 상세 ===
    zone_labels = ["미두(앞)", "미두~미상", "미상(중앙)", "미상~미미", "미미(꼬리)"]
    for i in range(5):
        dd = stu_dark[i] - ref_dark[i]
        d_dens = stu_dens[i] - ref_dens[i]
        
        if abs(dd) > 20:
            feedback.append({
                "type": "critical",
                "area": zone_labels[i],
                "weight": 8,
                "msg": f"{'심하게 진함' if dd > 0 else '심하게 연함'} (차이 {'+' if dd > 0 else ''}{dd:.0f})",
            })
            total_deduction += 8
        elif abs(dd) > 12:
            feedback.append({
                "type": "warning",
                "area": zone_labels[i],
                "weight": 5,
                "msg": f"{'다소 진함' if dd > 0 else '다소 연함'} (차이 {'+' if dd > 0 else ''}{dd:.0f})",
            })
            total_deduction += 5
        
        if abs(d_dens) > 20:
            feedback.append({
                "type": "warning",
                "area": zone_labels[i],
                "weight": 5,
                "msg": f"밀도 {'과다' if d_dens > 0 else '부족'} ({stu_dens[i]:.0f}% vs 정답 {ref_dens[i]:.0f}%)",
            })
            total_deduction += 5
    
    # 양호 항목
    if not any(f["area"] == "전체 레이어링" or f["area"] == "레이어링 부족" for f in feedback):
        if not any(f["area"] == "압/깊이 과다" or f["area"] == "약간 과다" for f in feedback):
            feedback.insert(0, {
                "type": "good",
                "area": "전체 명암",
                "weight": 0,
                "msg": "전체 명암이 정답과 유사합니다 ✨",
            })
    
    score = max(0, min(100, 100 - total_deduction))
    
    return {
        "score": score,
        "feedback": feedback,
        "problems": problems,
        "profile_80": profile_80,
        "zones_raw": [{"darkness": z["darkness"], "density": z["density"]} for z in zones],
    }


def detect_pattern(zones):
    """패턴 구분 (솔직 판정)"""
    profile_80 = normalize_to_80(zones)
    avg_dark = sum(profile_80) / 5
    
    # 레이어링 부족하면 패턴 구분 불가
    if avg_dark < 40:
        return {
            "pattern": "판별불가",
            "confidence": 0,
            "reason": "레이어링이 부족해서 패턴 특징이 나타나지 않습니다. 먼저 레이어링을 충분히 쌓은 후 다시 제출해주세요.",
        }
    
    scores = {}
    for name, ref in PATTERN_REFS.items():
        diff = sum(abs(zones[i]["darkness"] - ref["darkness"][i]) for i in range(5))
        dens_diff = sum(abs(zones[i]["density"] - ref["density"][i]) for i in range(5))
        scores[name] = diff + dens_diff * 0.5
    
    best = min(scores, key=scores.get)
    second = sorted(scores.values())[1]
    gap = second - scores[best]
    
    if gap < 10:
        return {
            "pattern": "판별불가",
            "confidence": 0,
            "reason": f"SOFT와 MIX가 비슷하게 나옵니다. 솔직히 구분이 어렵습니다. 어떤 패턴으로 작업했는지 다시 알려주시고, 어떤 점이 어려웠는지도 말씀해주세요.",
        }
    
    confidence = min(100, int(gap * 3))
    return {
        "pattern": best,
        "confidence": confidence,
        "reason": f"{best} 패턴으로 판단 (유사도 {confidence}%)",
    }


def analyze_image(image_path, pattern=None):
    """메인 분석 함수"""
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "이미지를 읽을 수 없습니다"}
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brows = detect_brows(image)
    
    if not brows:
        return {"error": "눈썹을 감지할 수 없습니다. 눈썹이 선명하게 보이는 사진으로 다시 보내주세요."}
    
    results = []
    for idx, brow in enumerate(brows):
        zones = analyze_zones(gray, brow)
        
        if pattern:
            pat = {"pattern": pattern, "confidence": 100, "reason": f"지정된 패턴: {pattern}"}
        else:
            pat = detect_pattern(zones)
        
        target_pattern = pat["pattern"] if pat["pattern"] in PATTERN_REFS else "SOFT"
        analysis = generate_feedback_v2(zones, target_pattern)
        
        results.append({
            "brow_index": idx + 1,
            "pattern": pat,
            "analysis": analysis,
        })
    
    # 공통 문제 추출
    if len(results) > 1:
        all_problems = [set(r["analysis"]["problems"]) for r in results]
        common = set.intersection(*all_problems) if all_problems else set()
        individual = [p - common for p in all_problems]
    else:
        common = set(results[0]["analysis"]["problems"]) if results else set()
        individual = [set()]
    
    return {
        "brow_count": len(brows),
        "results": results,
        "common_problems": list(common),
        "common_problem_descriptions": [COMMON_PROBLEMS.get(p, p) for p in common],
    }


def format_line_message(result, lang="ja"):
    """LINE 전송용 메시지 포맷"""
    if "error" in result:
        if lang == "ja":
            return f"❌ {result['error']}"
        return f"❌ {result['error']}"
    
    lines = []
    
    if lang == "ja":
        lines.append("ご提出ありがとうございます！🙇")
        lines.append("添削させていただきます。\n")
    else:
        lines.append("제출 감사합니다! 🙇")
        lines.append("첨삭 드립니다.\n")
    
    for r in result["results"]:
        a = r["analysis"]
        score = a["score"]
        pat = r["pattern"]
        
        if result["brow_count"] > 1:
            if lang == "ja":
                lines.append(f"━━ 眉{r['brow_index']} ({pat.get('pattern', '?')}) ━━")
            else:
                lines.append(f"━━ 눈썹{r['brow_index']} ({pat.get('pattern', '?')}) ━━")
        
        if lang == "ja":
            lines.append(f"スコア: {score}/100\n")
        else:
            lines.append(f"점수: {score}/100\n")
        
        # 양호
        good = [f for f in a["feedback"] if f["type"] == "good"]
        for f in good:
            lines.append(f"✨ {f['msg']}")
        
        # 문제
        problems = [f for f in a["feedback"] if f["type"] in ("critical", "warning")]
        if problems:
            if lang == "ja":
                lines.append("\n【改善ポイント】")
            else:
                lines.append("\n【개선 포인트】")
            
            for f in problems:
                icon = "🔴" if f["type"] == "critical" else "🟡"
                lines.append(f"{icon} {f['area']}: {f['msg']}")
        
        lines.append("")
    
    # 공통 문제
    if result.get("common_problems"):
        if lang == "ja":
            lines.append("【共通の課題】")
        else:
            lines.append("【공통 문제】")
        for desc in result["common_problem_descriptions"]:
            lines.append(f"📌 {desc}")
        lines.append("")
    
    if lang == "ja":
        lines.append("引き続き練習を頑張ってください！☺️")
    else:
        lines.append("계속 연습 화이팅! ☺️")
    
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = analyze_image(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("\n--- LINE Message (JA) ---")
        print(format_line_message(result, "ja"))
    else:
        print("Usage: python image_analyzer.py <image_path> [pattern]")
