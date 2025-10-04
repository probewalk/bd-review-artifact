import json

def dump_with_pairs_oneline(data, indent=2):
    """Custom serialization: keep normal indentation, but force "pairs" arrays into a single line"""
    def default(obj):
        return obj

    # Standard formatting
    text = json.dumps(data, indent=indent, ensure_ascii=False)

    # Replace each "pairs" field
    def fix_pairs(obj_text):
        obj = json.loads(obj_text)
        for key, value in obj.items():
            if isinstance(value, dict) and "pairs" in value:
                # Serialize "pairs" as one line
                pairs_str = '"pairs":' + json.dumps(value["pairs"], separators=(",", ":"))
                # Re-dump the rest of the object with indentation
                inner = value.copy()
                del inner["pairs"]
                inner_str = json.dumps(inner, indent=indent, ensure_ascii=False)
                # Concatenate
                fixed = inner_str[:-1] + ",\n" + " " * indent + pairs_str + "\n}"
                obj[key] = fixed  # Save as string
        return obj

    # Simple approach: traverse and handle specially
    new_obj = {}
    for k, v in data.items():
        if isinstance(v, dict) and "pairs" in v:
            pairs_str = json.dumps(v["pairs"], separators=(",", ":"))
            temp = v.copy()
            del temp["pairs"]
            block = json.dumps(temp, indent=indent, ensure_ascii=False)
            block = block[:-1] + ",\n" + " " * indent + f"\"pairs\":{pairs_str}\n" + "}"
            new_obj[k] = block
        else:
            new_obj[k] = v

    # Assemble top-level object
    lines = ["{"]
    for i, (k, v) in enumerate(new_obj.items()):
        comma = "," if i < len(new_obj) - 1 else ""
        if isinstance(v, str) and v.startswith("{"):
            # Already a formatted JSON string
            lines.append(" " * indent + f"\"{k}\": {v}{comma}")
        else:
            lines.append(" " * indent + f"\"{k}\": {json.dumps(v, ensure_ascii=False)}{comma}")
    lines.append("}")
    return "\n".join(lines)


if __name__ == "__main__":
    with open("./data/real-data/info.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    result = dump_with_pairs_oneline(data, indent=2)

    with open("./data/real-data/info_fixed.json", "w", encoding="utf-8") as f:
        f.write(result)
