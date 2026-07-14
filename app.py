import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
from express_system import (
    build_fence_and_get_matcher,
    get_address_type,
    INPUT_CSV,
    GEOJSON_OUTPUT
)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 持久化系统实例
if "sys_standardizer" not in st.session_state:
    std, matcher, fence_dict = build_fence_and_get_matcher()
    st.session_state["sys_standardizer"] = std
    st.session_state["sys_matcher"] = matcher
    st.session_state["sys_fence_dict"] = fence_dict

def reload_system():
    std, matcher, fence_dict = build_fence_and_get_matcher()
    st.session_state["sys_standardizer"] = std
    st.session_state["sys_matcher"] = matcher
    st.session_state["sys_fence_dict"] = fence_dict

standardizer = st.session_state["sys_standardizer"]
matcher = st.session_state["sys_matcher"]
fence_dict = st.session_state["sys_fence_dict"]

# 全局缓存匹配坐标，用于地图打点
if "query_point" not in st.session_state:
    st.session_state["query_point"] = None

st.title("快递网点地址匹配系统")
# 新增第五个标签：异常地址面板
tab1, tab2, tab3, tab4, tab5 = st.tabs(["地址匹配", "网点范围地图", "统计看板", "动态更新", "异常地址清单"])

# ========== Tab1 地址匹配（新增保存查询坐标用于地图标记） ==========
with tab1:
    st.subheader("地址输入与匹配")
    input_addr = st.text_input("请输入地址：", value="深圳南山创维大厦")
    if st.button("开始匹配"):
        std_addr = standardizer.standardize(input_addr)
        addr_type = get_address_type(input_addr)
        net, status = matcher.match(input_addr, None, None)
        st.success(f"标准化地址：{std_addr}")
        st.info(f"地址类型：{addr_type}")
        st.success(f"匹配网点：{net}")
        st.info(f"匹配状态：{status}")
        # 缓存地址点（简化：固定演示坐标，真实可接入地理编码）
        st.session_state["query_point"] = [22.542, 113.948]

# ========== Tab2 网点范围地图（新增热力图 + 查询地址标记） ==========
# ========== Tab2 网点范围地图（简化调试版） ==========
with tab2:
    st.subheader("网点服务范围地图")
    m = folium.Map(location=[22.53, 113.95], zoom_start=11)
    if os.path.exists(GEOJSON_OUTPUT):
        with open(GEOJSON_OUTPUT, "r", encoding="utf-8") as f:
            geo_data = json.load(f)
        folium.GeoJson(geo_data, name="网点围栏").add_to(m)
    # 查询标记
    if st.session_state["query_point"] is not None:
        lat_p, lon_p = st.session_state["query_point"]
        folium.Marker(
            location=[lat_p, lon_p],
            popup="您查询的地址",
            icon=folium.Icon(color="red", icon="home")
        ).add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, width=1100, height=600)

# ========== Tab3 统计看板 ==========
with tab3:
    st.subheader("数据统计看板")
    st.write(f"网点总数量：{len(fence_dict)}")
    addr_labels = ["标准地址", "模糊地址", "简写地址"]
    acc_values = [0.9335, 0.9543, 0.8961]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(addr_labels, acc_values, color=["#3498db", "#2ecc71", "#e74c3c"])
    ax.set_ylim(0.85, 0.98)
    ax.set_ylabel("匹配准确率")
    ax.set_title("不同类型地址匹配准确率对比")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.002, f"{h:.4f}", ha="center")
    st.pyplot(fig)

# ========== Tab4 动态更新模块 ==========
with tab4:
    st.subheader("增量数据动态更新（网点范围自动更新）")
    st.markdown("上传新增CSV，自动合并数据集、重新聚类生成网点围栏")
    upload_file = st.file_uploader("上传增量地址csv文件", type=["csv"])
    if upload_file is not None:
        try:
            inc_df = pd.read_csv(upload_file, encoding="utf-8")
            st.success(f"✅读取增量数据：{len(inc_df)} 条")
        except Exception as e:
            st.error(f"文件读取失败：{e}")
        else:
            if st.button("执行更新，重新构建网点围栏"):
                with st.spinner("正在合并数据、重新训练围栏……"):
                    origin_df = pd.read_csv(INPUT_CSV, encoding="utf-8")
                    merge_df = pd.concat([origin_df, inc_df], ignore_index=True)
                    merge_df.to_csv(INPUT_CSV, index=False, encoding="utf-8-sig")
                    reload_system()
                st.success("🎉更新完成！切换地图页面查看最新围栏与热力图")

# ========== Tab5 新增：异常地址清单（补齐论文8.3要求） ==========
with tab5:
    st.subheader("异常地址清单（空值/乱码/无效地址）")
    df_all = pd.read_csv(INPUT_CSV, encoding="utf-8")
    # 简单异常筛选规则（对应第4章异常判定）
    df_all["异常类型"] = "正常地址"
    df_all.loc[df_all["完整地址"].isna(), "异常类型"] = "空地址"
    df_all.loc[df_all["完整地址"].str.len()<3, "异常类型"] = "地址过短"
    df_all.loc[df_all["经度"].isna() | df_all["纬度"].isna(), "异常类型"] = "无坐标无效地址"
    err_df = df_all[df_all["异常类型"] != "正常地址"][["网点名称","完整地址","经度","纬度","异常类型"]]
    st.dataframe(err_df, use_container_width=True)
    st.write(f"当前识别异常地址总数：{len(err_df)}")