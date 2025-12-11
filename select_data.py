import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from collections import defaultdict

# ======================
# 1. 读取并清洗数据
# ======================
@st.cache_data
def load_data(file):
    # file 可以是文件名字符串，也可以是上传的file对象
    df = pd.read_excel(file, skiprows=1)

    angle_cols = ['夹角角度min', '夹角角度max', '工作角min', '工作角max', '前进角min', '前进角max']
    for col in angle_cols:
        s = df[col].astype(str).str.replace(r'[°\s]', '', regex=True).replace('', '-1')
        df[col] = pd.to_numeric(s, errors='coerce').fillna(-1).astype(int)

    key_cols = ['板材', '夹角角度min', '夹角角度max', '工作角min', '工作角max', '前进角min', '前进角max']
    other_cols = [col for col in df.columns if col not in key_cols]
    df = df[key_cols + other_cols]
    return df



# ======================
# 2. 页面标题
# ======================
st.title("工艺参数交集分析工具")
st.caption("✅ 1°精度逐点验证 | ✅ 单点规则不泛化 | ✅ 多板材交集")

# ======================
# 2.1 选择数据文件
# ======================
st.sidebar.header("📁 数据文件")

uploaded_file = st.sidebar.file_uploader(
    "选择 Excel 文件",
    type=["xlsx", "xls"],
    help="如不选择，则使用默认文件：在线验证汇总表.xlsx"
)

default_file = "在线验证汇总表.xlsx"

if uploaded_file is not None:
    data_source = uploaded_file
    st.sidebar.success(f"已加载：{uploaded_file.name}")
else:
    data_source = default_file
    st.sidebar.info(f"当前使用默认文件：{default_file}")

# 这里再真正读数据
df = load_data(data_source)

# ======================
# 3. 显示原始数据
# ======================
with st.expander("🔍 查看完整原始数据（关键列靠左）"):
    st.dataframe(df, width='content', height=400)

# ======================
# 4. 用户输入（优化交互）
# ======================
st.sidebar.header("🔧 筛选条件")

materials = st.sidebar.multiselect(
    "选择板材",
    options=sorted(df['板材'].dropna().unique()),
    default=sorted(df['板材'].dropna().unique())[:2]
)

# 夹角区间管理
if 'angle_ranges' not in st.session_state:
    st.session_state.angle_ranges = [(60, 90), (90, 120)]

st.sidebar.write("### 夹角范围设置")
for i, (a, b) in enumerate(st.session_state.angle_ranges):
    cols = st.sidebar.columns([3, 3, 1])
    new_a = cols[0].number_input(f"Min {i+1}", value=a, min_value=0, max_value=180, key=f"min_{i}")
    new_b = cols[1].number_input(f"Max {i+1}", value=b, min_value=0, max_value=180, key=f"max_{i}")
    if cols[2].button("×", key=f"del_{i}"):
        st.session_state.angle_ranges.pop(i)
        st.rerun()
    st.session_state.angle_ranges[i] = (min(new_a, new_b), max(new_a, new_b))

if st.sidebar.button("➕ 添加夹角区间"):
    st.session_state.angle_ranges.append((0, 180))
    st.rerun()

target_ranges = st.session_state.angle_ranges
update_time = datetime.now().strftime("%H:%M:%S")

# ======================
# 5. 核心逻辑：按夹角逐度求交集（1°精度）
# ======================

def get_angle_degrees_from_ranges(ranges):
    """从用户设定的夹角区间生成要遍历的所有角度值（整数，去重）"""
    angle_set = set()
    for amin, amax in ranges:
        if amin is None or amax is None:
            continue
        for ang in range(int(amin), int(amax) + 1):
            angle_set.add(ang)
    return sorted(angle_set)

def enumerate_pairs_from_rows(rows):
    """
    给定若干行（同一板材、同一夹角下满足条件的行），
    在 1° 精度下展开这些行对应的所有 (工作角, 前进角) 组合，并做并集。
    """
    pairs = set()
    for _, r in rows.iterrows():
        w_min = int(r['工作角min'])
        w_max = int(r['工作角max'])
        a_min = int(r['前进角min'])
        a_max = int(r['前进角max'])

        # 略过无效行
        if w_min < 0 or w_max < 0 or a_min < 0 or a_max < 0:
            continue
        if w_min > w_max or a_min > a_max:
            continue

        for w in range(w_min, w_max + 1):
            for a in range(a_min, a_max + 1):
                pairs.add((w, a))
    return pairs
def merge_to_ranges(values):
    """把一堆离散整数合并成若干连续区间 [(s1,e1), (s2,e2), ...]"""
    vals = sorted(set(values))
    if not vals:
        return []
    ranges = []
    start = prev = vals[0]
    for v in vals[1:]:
        if v == prev + 1:
            prev = v
        else:
            ranges.append((start, prev))
            start = prev = v
    ranges.append((start, prev))
    return ranges

def format_ranges(ranges):
    """[(s,e)] -> 's°~e° ∪ s2°~e2°'"""
    parts = []
    for s, e in ranges:
        if s == e:
            parts.append(f"{s}°")
        else:
            parts.append(f"{s}°~{e}°")
    return " ∪ ".join(parts) if parts else "无"

st.subheader("✅ 公共可行工艺参数（1°精度）")

if not materials or not target_ranges:
    st.info("请在左侧选择板材并设置夹角范围")
else:
    # 1. 先根据夹角区间生成需要遍历的所有夹角角度值
    target_angles = get_angle_degrees_from_ranges(target_ranges)
    if not target_angles:
        st.warning("⚠️ 当前夹角区间设置无有效角度，请调整区间")
    else:
        material_feasible = {}  # 每个板材对应一套可行 (w, a) 组合（已在所有夹角上取过交集）

        # 2. 对每个板材，按“夹角逐度 + 行内并集 + 夹角间交集”的方式计算
        for material in materials:
            df_mat = df[df['板材'] == material]
            if df_mat.empty:
                continue

            feasible_for_all_angles = None  # 该板材在所有夹角上的交集结果

            for ang in target_angles:
                # 2.1 找出当前夹角 ang 下，满足 夹角min <= ang <= 夹角max 的行
                rows = df_mat[(df_mat['夹角角度min'] <= ang) & (df_mat['夹角角度max'] >= ang)]

                # 如果该角度下，没有任何规则可用，则该板材在所有夹角上的交集必然为空
                if rows.empty:
                    feasible_for_all_angles = set()
                    break

                # 2.2 在 1° 粒度下，展开当前角度 ang 下所有可行的 (工作角, 前进角) 组合（并集）
                pairs_at_angle = enumerate_pairs_from_rows(rows)

                # 如果这个角度下压根没有可行组合，同样交集为空
                if not pairs_at_angle:
                    feasible_for_all_angles = set()
                    break

                # 2.3 对所有夹角做交集
                if feasible_for_all_angles is None:
                    feasible_for_all_angles = pairs_at_angle
                else:
                    feasible_for_all_angles &= pairs_at_angle

                # 如果交集已经为空，可以提前结束
                if not feasible_for_all_angles:
                    break

            # 该板材在所有目标夹角下的可行组合
            if feasible_for_all_angles:
                material_feasible[material] = feasible_for_all_angles

        # 3. 如果所有板材都没有可行组合
        if not material_feasible:
            st.warning(f"⚠️ 未找到满足所有夹角的工艺参数（更新时间：{update_time}）")
        else:
            # 4. 多板材之间做交集
            all_feasible = None
            for material, pairs in material_feasible.items():
                if all_feasible is None:
                    all_feasible = set(pairs)
                else:
                    all_feasible &= pairs

            if not all_feasible:
                st.error("❌ 在所有选中板材和所有目标夹角下，无公共 (工作角, 前进角) 组合")
            else:
                # ========= 先根据 all_feasible 做分组 =========
                grouped_by_work = defaultdict(list)
                grouped_by_adv  = defaultdict(list)

                for w, a in sorted(all_feasible):
                    grouped_by_work[w].append(a)
                    grouped_by_adv[a].append(w)
                # ========= 公共前进角：对所有工作角取交集 =========
                adv_common = None
                for w, adv_list in grouped_by_work.items():
                    s = set(adv_list)
                    if adv_common is None:
                        adv_common = s
                    else:
                        adv_common &= s

                adv_common = adv_common or set()  # 防止 None
                adv_common_ranges = merge_to_ranges(adv_common)

                # ========= 公共工作角：对所有前进角取交集（同理） =========
                work_common = []
                # 遍历工作角30-60，work_i，遍历adv_common_ranges内的每个adv_i，如果work_i在grouped_by_adv[adv_i]的范围内，则work_i是公共工作角
                for w in range(30, 61):
                    #遍历adv_common_ranges内的每个adv_i
                    for adv_range in adv_common_ranges:
                        for adv_i in range(adv_range[0], adv_range[1]+1):
                            if w in grouped_by_adv[adv_i]:
                                work_common.append(w)
                                break
                work_common_ranges = merge_to_ranges(work_common)

                # ========= 摘要展示 =========
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"**公共工作角范围**: {format_ranges(work_common_ranges)}")
                with col2:
                    st.success(f"**公共前进角范围**: {format_ranges(adv_common_ranges)}")

                st.caption(f"🕒 结果更新时间：{update_time} | 共 {len(all_feasible)} 个可行组合")


                # 5. 可视化
                st.subheader("📊 可行组合分布")
                w_list, a_list = zip(*all_feasible)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=w_list, y=a_list,
                    mode='markers',
                    marker=dict(size=3, color='green'),
                    name='可行组合'
                ))
                fig.update_layout(
                    xaxis_title="工作角 (°)",
                    yaxis_title="前进角 (°)",
                    height=450,
                    xaxis=dict(range=[-5, 185]),
                    yaxis=dict(range=[-5, 185])
                )
                st.plotly_chart(fig, use_container_width=True)

                # 6. 详细列表（按工作角分组 + 连续前进角合并）
                with st.expander("🔍 查看详细可行组合（按工作角分组）"):
                    from collections import defaultdict
                    grouped = defaultdict(list)
                    for w, a in sorted(all_feasible):
                        grouped[w].append(a)

                    detail_rows = []
                    for w in sorted(grouped):
                        adv_list = sorted(grouped[w])
                        # 合并连续的前进角成区间
                        ranges = []
                        start = end = adv_list[0]
                        for val in adv_list[1:]:
                            if val == end + 1:
                                end = val
                            else:
                                ranges.append(f"{start}~{end}" if start != end else str(start))
                                start = end = val
                        ranges.append(f"{start}~{end}" if start != end else str(start))

                        detail_rows.append({
                            "工作角 (°)": w,
                            "前进角可用值": " ∪ ".join(ranges)
                        })

                    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)
    
    
    
    #python -m streamlit run select_data.py