import sys, json
p = sys.argv[1]
with open(p, 'r', encoding='utf-8') as f:
    obj = json.load(f)

# If list, show first item
if isinstance(obj, list):
    item = obj[0]
else:
    # if dict with 'data' or similar
    if isinstance(obj, dict):
        # try to find list under common keys
        for k in ('data','conversations','examples','items'):
            if k in obj and isinstance(obj[k], list):
                print('top_key:', k)
                item = obj[k][0]
                break
        else:
            item = obj
    else:
        item = obj

print('type:', type(obj), 'item sample keys:', list(item.keys()) if isinstance(item, dict) else str(type(item)))
print('sample:', json.dumps(item, indent=2)[:1000])
