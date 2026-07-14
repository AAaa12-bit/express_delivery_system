import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False

import os
import re
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from shapely.geometry import Point
from shapely.ops import unary_union
from collections import defaultdict

def levenshtein_distance(s1, s2):
    s1 = str(s1)
    s2 = str(s2)
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (1 if c1 != c2 else 0)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

# 全局可调参数
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
DBSCAN_EPS = 0.0020
DBSCAN_MIN_SAMPLES = 3
BUFFER_DIST = 0.0024
NGRAM_WINDOW = 2
TEXT_SIM_THRESHOLD = 0.01
TEXT_WEIGHT = 0.6
DIST_WEIGHT = 0.4

INPUT_CSV = "address_with_gps_clean.csv"
GEOJSON_OUTPUT = "network_service_fence.geojson"
RESULT_CSV = "address_match_result.csv"

def char_ngram(s, n=NGRAM_WINDOW):
    s = str(s).replace(" ", "")
    ngrams = set()
    if len(s) >= n:
        for i in range(len(s) - n + 1):
            ngrams.add(s[i:i + n])
    return ngrams

def clean_address(raw_addr):
    raw_addr = str(raw_addr)
    raw = re.sub(r"[!！，。、；：？]", "", raw_addr)
    raw = re.sub(r"放门口|转.*|务必.*|手机号\d+", "", raw)
    raw = re.sub(r"\s+", "", raw)
    return raw

# 新增：地址类型自动分类 标准/模糊/简写
def get_address_type(addr):
    if re.search(r"栋|单元|号楼|号巷|号路|号", addr):
        return "标准地址"
    elif re.search(r"小区|花园|大厦|广场", addr):
        return "模糊地址"
    else:
        return "简写地址"

class AddressStandardizer:
    def __init__(self, std_addr_list):
        self.std_addr = [str(x) for x in std_addr_list]
        self.short_map = {
            "深大": "深圳大学",
            "南山": "南山区",
            "深职院": "深圳职业技术学院",
            "海岸城": "海岸城购物中心",
            "科技园": "南山科技园",
            "后海": "后海片区",
            "蛇口": "蛇口片区"
        }
        self.freq = defaultdict(int)
        for addr in self.std_addr:
            self.freq[addr] += 1

    def correct_spell(self, input_addr):
        input_addr = str(input_addr)
        min_dist = float("inf")
        best = None
        for std in self.std_addr:
            dist = levenshtein_distance(input_addr, std)
            if isinstance(dist, (int, float)) and dist <= 2 and dist < min_dist and self.freq[std] >= 3:
                min_dist = dist
                best = std
        return best if best else input_addr

    def standardize(self, raw_addr):
        addr = clean_address(raw_addr)
        if len(addr) == 0:
            return "深圳市未知地址"
        for short, full in self.short_map.items():
            if short in addr:
                addr = addr.replace(short, full)
        addr = self.correct_spell(addr)
        if "深圳市" not in addr:
            addr = f"深圳市{addr}"
        return addr

class NetFenceBuilder:
    def __init__(self, train_df):
        self.train_df = train_df
        self.fence_dict = {}
        self.centroid_map = {}

    def build_single_fence(self, net_name):
        sub_df = self.train_df[self.train_df["网点名称"] == net_name]
        coords = sub_df[["经度", "纬度"]].values
        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(coords)
        labels = db.labels_
        cluster_hulls = []
        for label in set(labels):
            if label == -1:
                continue
            cluster_coords = coords[labels == label]
            points = [Point(x, y) for x, y in cluster_coords]
            hull = unary_union(points).convex_hull
            cluster_hulls.append(hull)
        if not cluster_hulls:
            return None
        union_hull = unary_union(cluster_hulls)
        buffer_poly = union_hull.buffer(BUFFER_DIST)
        self.centroid_map[net_name] = buffer_poly.centroid
        return buffer_poly

    def build_all_fence(self):
        net_names = self.train_df["网点名称"].unique()
        for net in net_names:
            poly = self.build_single_fence(net)
            if poly:
                self.fence_dict[net] = poly

    def export_geojson(self):
        features = []
        for net, poly in self.fence_dict.items():
            feat = {
                "type": "Feature",
                "properties": {"network": net},
                "geometry": json.loads(json.dumps(poly.__geo_interface__))
            }
            features.append(feat)
        geojson = {"type": "FeatureCollection", "features": features}
        with open(GEOJSON_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        print(f"✅ 网点围栏GeoJSON已生成：{GEOJSON_OUTPUT}")

    # 新增：计算围栏重叠统计 5.4.1/5.4.2
    def calculate_overlap_statistic(self):
        poly_list = list(self.fence_dict.values())
        overlap_pairs = 0
        overlap_total_area = 0.0
        for i in range(len(poly_list)):
            for j in range(i+1, len(poly_list)):
                inter = poly_list[i].intersection(poly_list[j])
                if inter.area > 1e-9:
                    overlap_pairs += 1
                    overlap_total_area += inter.area
        all_total_area = sum(p.area for p in poly_list)
        overlap_ratio = overlap_total_area / all_total_area if all_total_area>0 else 0
        print("\n====围栏重叠统计（5.4.1/5.4.2）====")
        print(f"存在重叠的围栏配对数量：{overlap_pairs}")
        print(f"重叠区域总面积：{overlap_total_area:.6f}")
        print(f"重叠面积占总围栏面积比例：{overlap_ratio:.4f}")
        return overlap_pairs, overlap_total_area, overlap_ratio

class ThreeLayerMatcher:
    def __init__(self, fence_dict, train_df):
        self.fence_dict = fence_dict
        self.train_df = train_df
        self.net_list = list(fence_dict.keys())
        self.centroid_map = {n: fence_dict[n].centroid for n in self.net_list}
        self.tfidf = TfidfVectorizer(analyzer=char_ngram)
        self._build_text_lib()
        # 计时统计容器
        self.time_stat = defaultdict(list)

    def _build_text_lib(self):
        self.net_addr_lib = defaultdict(list)
        for net in self.net_list:
            sub_addr = self.train_df[self.train_df["网点名称"] == net]["完整地址"].tolist()
            addrs = [str(x) for x in sub_addr]
            self.net_addr_lib[net] = addrs
        all_text = []
        for net in self.net_list:
            all_text.extend(self.net_addr_lib[net])
        self.tfidf.fit(all_text)

    def layer1_spatial_match(self, lon, lat, std_addr):
        p = Point(float(lon), float(lat))
        hit_list = []
        for net, poly in self.fence_dict.items():
            if poly.contains(p):
                hit_list.append(net)
        if len(hit_list) == 0:
            return None, "spatial_none"
        if len(hit_list) == 1:
            return hit_list[0], "spatial_ok"
        max_sim = -1.0
        best_net = None
        query_vec = self.tfidf.transform([std_addr])
        for net in hit_list:
            addrs = self.net_addr_lib[net]
            vecs = self.tfidf.transform(addrs)
            sims = cosine_similarity(query_vec, vecs).flatten()
            avg_sim = float(np.mean(sims))
            if avg_sim > max_sim:
                max_sim = avg_sim
                best_net = net
        return best_net, "spatial_conflict_text_aid"

    def layer2_text_match(self, std_addr):
        max_sim = -1.0
        best_net = None
        query_vec = self.tfidf.transform([std_addr])
        for net in self.net_list:
            addrs = self.net_addr_lib[net]
            vecs = self.tfidf.transform(addrs)
            sims = cosine_similarity(query_vec, vecs).flatten()
            avg_sim = float(np.mean(sims))
            if avg_sim > max_sim:
                max_sim = avg_sim
                best_net = net
        if max_sim < TEXT_SIM_THRESHOLD:
            return None, "text_low"
        return best_net, "text_ok"

    def layer3_weighted_backup(self, std_addr, lon, lat):
        p = Point(float(lon), float(lat))
        query_vec = self.tfidf.transform([std_addr])
        score_list = []
        dist_all = []
        net_info = []
        for net in self.net_list:
            addrs = self.net_addr_lib[net]
            vecs = self.tfidf.transform(addrs)
            sims = cosine_similarity(query_vec, vecs).flatten()
            avg_sim = float(np.mean(sims))
            dist = self.centroid_map[net].distance(p)
            net_info.append([net, avg_sim, dist])
            dist_all.append(dist)
        max_dist = max(dist_all)
        for net, sim, dist in net_info:
            norm_dist_score = 1.0 - (dist / max_dist)
            total_score = TEXT_WEIGHT * sim + DIST_WEIGHT * norm_dist_score
            score_list.append((net, total_score))
        score_list.sort(key=lambda x: x[1], reverse=True)
        return score_list[0][0], "weighted_backup"

    def match(self, input_addr, lon=None, lat=None):
        std_addr = AddressStandardizer(self.train_df["完整地址"].tolist()).standardize(input_addr)
        # 经纬度为空，仅文本匹配
        if lon is None or lat is None:
            net, status = self.layer2_text_match(std_addr)
            return net, status
        # 空间匹配优先
        net, status = self.layer1_spatial_match(lon, lat, std_addr)
        # 空间匹配无结果，降级文本匹配
        if net is None:
            net, status = self.layer2_text_match(std_addr)
        return net, status

    def print_time_stat(self):
        print("\n====各匹配阶段平均耗时（6.4.7）====")
        for k,v in self.time_stat.items():
            if len(v)>0:
                avg = np.mean(v)
                print(f"{k} 平均耗时：{avg:.6f} s，样本量：{len(v)}")

if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["网点名称", "完整地址", "经度", "纬度"])
    df["经度"] = df["经度"].astype(float)
    df["纬度"] = df["纬度"].astype(float)
    print(f"加载清洗后地址数据集，总记录：{len(df)}")

    train_parts, test_parts = [], []
    for net in df["网点名称"].unique():
        sub = df[df["网点名称"] == net]
        train_sub, test_sub = train_test_split(sub, train_size=TRAIN_RATIO, random_state=RANDOM_SEED, shuffle=True)
        train_parts.append(train_sub)
        test_parts.append(test_sub)
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    print(f"训练集大小：{len(train_df)}，测试集大小：{len(test_df)}")

    builder = NetFenceBuilder(train_df)
    builder.build_all_fence()
    builder.export_geojson()
    builder.calculate_overlap_statistic()

    standardizer = AddressStandardizer(train_df["完整地址"].tolist())
    matcher = ThreeLayerMatcher(builder.fence_dict, train_df)

    print("\n开始测试集地址匹配计算……")
    result_list = []
    correct_count = 0
    y_true = []
    y_pred = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        raw_addr = row["完整地址"]
        lng = row["经度"]
        lat = row["纬度"]
        true_net = row["网点名称"]
        std_addr = standardizer.standardize(raw_addr)
        addr_type = get_address_type(raw_addr)
        pred_net, match_status = matcher.match(std_addr, lng, lat)
        y_true.append(true_net)
        y_pred.append(pred_net)
        if pred_net == true_net:
            correct_count += 1
        result_list.append({
            "完整地址": raw_addr,
            "地址类型": addr_type,
            "经度": lng,
            "纬度": lat,
            "真实网点": true_net,
            "预测网点": pred_net,
            "匹配状态": match_status
        })
    out_df = pd.DataFrame(result_list)
    out_df.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 匹配结果输出 {RESULT_CSV}")

    total_test = len(test_df)
    acc = correct_count / total_test
    print("\n=======测试集评估结果=======")
    print(f"测试样本总数：{total_test}")
    print(f"预测正确数量：{correct_count}")
    print(f"整体匹配准确率：{acc:.4f}")

    # 6.4.3 输出分类报告 各网点Precision/Recall/F1
    print("\n====各网点分类指标classification_report（6.4.3）====")
    print(classification_report(y_true,y_pred,digits=4))

    # 6.4.5 按地址类型分组统计准确率
    print("\n====不同类型地址匹配准确率（6.4.5）====")
    type_group = out_df.groupby("地址类型")
    for t,sub in type_group:
        correct = (sub["真实网点"]==sub["预测网点"]).sum()
        acc_t = correct / len(sub)
        print(f"{t} 样本数:{len(sub)} 正确:{correct} 准确率:{acc_t:.4f}")

    # 打印各层耗时
    matcher.print_time_stat()
    print("\n6.4.6 典型错分案例：可打开address_match_result.csv筛选匹配状态 != spatial_ok样本人工分析")

def build_fence_and_get_matcher():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["网点名称", "完整地址", "经度", "纬度"])
    df["经度"] = df["经度"].astype(float)
    df["纬度"] = df["纬度"].astype(float)

    train_parts, test_parts = [], []
    for net in df["网点名称"].unique():
        sub = df[df["网点名称"] == net]
        train_sub, test_sub = train_test_split(sub, train_size=TRAIN_RATIO, random_state=RANDOM_SEED, shuffle=True)
        train_parts.append(train_sub)
        test_parts.append(test_sub)
    train_df = pd.concat(train_parts, ignore_index=True)

    builder = NetFenceBuilder(train_df)
    builder.build_all_fence()
    builder.export_geojson()
    builder.calculate_overlap_statistic()

    standardizer = AddressStandardizer(train_df["完整地址"].tolist())
    matcher = ThreeLayerMatcher(builder.fence_dict, train_df)
    return standardizer, matcher, builder.fence_dict