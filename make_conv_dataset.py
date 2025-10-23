from typing import List, Dict, Any

def conversations_to_samples(conversations_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    주어진 대화 객체 {"conversations": [{role, content}, {role, content}, ...]}를 학습 샘플 리스트로 변환한다.
    
    규칙:
      - i번째 메시지가 role_1, i+1번째가 role_2면 1개 샘플 생성
      - history = conversations[0 : i+1]  (0~i까지 포함)
      - response = conversations[i+1]     (role_2 1개)
    반환 형식:
      [
        {
          "history": [ {role, content}, ..., {role, content} ],
          "response": { "role": "role_2", "content": "..." }
        },
        ...
      ]
    """
    if "conversations" not in conversations_obj:
        raise ValueError("Input dict must have a 'conversations' key.")

    convs = conversations_obj["conversations"]
    if not isinstance(convs, list):
        raise TypeError("'conversations' must be a list of messages.")

    # 유효성 검사
    for idx, msg in enumerate(convs):
        if not isinstance(msg, dict):
            raise TypeError(f"Message at index {idx} must be a dict.")
        if "role" not in msg or "content" not in msg:
            raise ValueError(f"Message at index {idx} must have 'role' and 'content'.")
        if msg["role"] not in {"role_1", "role_2"}:
            raise ValueError(f"Unexpected role at index {idx}: {msg['role']}")
        if not isinstance(msg["content"], str) or msg["content"].strip() == "":
            raise ValueError(f"'content' must be non-empty string at index {idx}.")

    samples: List[Dict[str, Any]] = []

    # i: role_1, i+1: role_2 인 페어만 수집
    for i in range(len(convs) - 1):
        cur_msg = convs[i]
        nxt_msg = convs[i + 1]

        if cur_msg["role"] == "role_1" and nxt_msg["role"] == "role_2":
            history = convs[: i + 1]             # 0 ~ i (role_1까지)
            response = nxt_msg                   # i+1 (role_2 1개)
            sample = {
                "history": history,
                "response": response,
            }
            samples.append(sample)

    return samples



if __name__ == "__main__":
    import json
    
    input_file = "###"
    output_file = "###"
    with open(input_file, "r", encoding="utf-8") as in_f, open(output_file, "w", encoding="utf-8") as out_f:
        lines = in_f.readlines()
        
        # 마지막 발화가 role_1인 경우와 role_2인 경우 카운트
        last_is_role_1 = 0
        last_is_role_2 = 0 
        
        for line in lines:
            line_json = json.loads(line)
            location = line_json.get("location", "unknown_location")
            feature1 = line_json.get("feature1", "unknown_feature1")
            feature2 = line_json.get("feature2", "unknown_feature2")
            code = line_json.get("code", "unknown_code")

            if line_json["conversations"][-1].get("role") != "role_2":
                last_is_role_1 += 1
                continue
            else:
                last_is_role_2 += 1
            conv_data = {"conversations": line_json["conversations"]}
            samples = conversations_to_samples(conv_data)

            for k, s in enumerate(samples, start=1):
                print(f"[{k}번 샘플]")
                s["location"] = location
                s["feature1"] = feature1
                s["feature2"] = feature2
                s["code"] = code
                s_json = json.dumps(s, ensure_ascii=False, indent=4)
                out_f.write(s_json)
                out_f.write("\n")
        
        print(last_is_role_1)
        print(last_is_role_2)