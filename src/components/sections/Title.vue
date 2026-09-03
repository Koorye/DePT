<script lang="ts" setup>

import { ElIcon } from 'element-plus'
import { Link, Document } from '@element-plus/icons-vue'

// logo地址，没有则置为""即可
const logo = ''

// 标题
const title = 'DePT: Decoupled Prompt Tuning'

// 标题颜色
const title_color = '#000000'

// 标题补充，没有则置为''即可
const title_supp = ''

// 作者清单（包含作者姓名、头像、主页、地址序号）
const authors = [
  {
    name: "Shihan Wu",
    homepage: "https://Koorye.github.io/",
    address_flag: "1*"
  },
  {
    name: "Lianli Gao",
    homepage: "https://github.com",
    address_flag: "2"
  },
  {
    name: "Heng Tao Shen",
    homepage: "https://github.com",
    address_flag: "3"
  },
  {
    name: "Jingkuan Song",
    homepage: "https://github.com",
    address_flag: "3,1†"
  },
]

// 地址清单（包含地址名称、头像、主页、地址序号）
const addresses = [
  {
    address_flag: "1",
    name: "University of Electronic Science and Technology of China",
    homepage: "https://github.com",
    // icon: "./icons/uestc.jpg"
  },
  {
    address_flag: "2",
    name: "Shenzhen Institute for Advanced Study, UESTC",
    homepage: "https://github.com",
    // icon: "./icons/sjtu.jpg"
  },
  {
    address_flag: "3",
    name: "Tongji University",
    homepage: "https://github.com",
    // icon: "./icons/tju.jpg"
  },
]

// 共一和通讯提示
const con_and_corresponding_author = "*Equal Contribution  †Corresponding author"

const news = [
  "🔥Feb 27, 2024: 🎉🎉Our paper has been accepted by CVPR 2024!🎉🎉",
  "🔥Nov 5, 2023: Training and evaluation codes for DePT are released.",
  "🔥Sep 14, 2023: The paper is now available on arXiv."
]

// 强调内容
const emphases = [
  "🎉🎉Our paper has been accepted by CVPR 2024!🎉🎉"
]

// 提供引导资料链接
const buttons = [
  {
    disabled: false,
    name: "ArXiv",
    link: "https://arxiv.org/abs/2309.07439",
    component: Link,
  },
  {
    disabled: false,
    name: "PDF",
    link: "https://arxiv.org/pdf/2309.07439",
    component: Document,
  },
  {
    disabled: false,
    name: "Code",
    link: "https://github.com/Koorye/DePT",
    component: Link,
  }
]

</script>

<template>
  <div class="hero">

    <!-- 最新消息提示 -->
    <el-row justify="center" class="news-list">
      <el-col :xs="28" :sm="24" :md="20" :lg="16" :xl="14">
        <div
          class="news-chip"
          v-for="(item, index) in news"
          :key="index"
          :class="{ highlight: index === 0 }"
        >
          <span class="news-dot"></span>
          <span class="news-text">{{ item }}</span>
        </div>
      </el-col>
    </el-row>

    <!-- 文章logo -->
    <el-row v-if="logo" justify="center">
      <el-image :src="logo" class="logo" fit="cover" />
    </el-row>

    <!-- 文章标题 -->
    <el-row justify="center">
      <el-col :span="20">
        <h1 class="paper-title">
          <div v-if="title" :style="{color:title_color}"> {{ title }}</div>
          <div v-if="title_supp" class="title-supp"> {{ title_supp }}</div>
        </h1>
      </el-col>
    </el-row>

    <!-- 作者名单 -->
    <el-row justify="center">
      <a :href=author.homepage v-for="author in authors">
        <el-button class="title-button" type="primary" text>
          <span class="author">
            {{ author.name }}<sup v-if="author.address_flag" class="name_sup">{{ author.address_flag }}</sup>
          </span>
        </el-button>
      </a>
    </el-row>

    <!-- 地址名单 -->
    <el-row justify="center">
      <a :href=address.homepage v-for="address in addresses">
        <el-button class="title-button" type="primary" text>
          <el-avatar v-if="address.icon" :size="40" :src="address.icon" />
          <span class="address">
            <sup v-if="address.address_flag" class="address_sup">{{ address.address_flag }}</sup>{{ address.name }}
          </span>
        </el-button>
      </a>
    </el-row>

    <!-- 共一和通讯提示内容 -->
    <el-row justify="center" class="con-cor">
        {{ con_and_corresponding_author }}
    </el-row>

    <!-- 强调内容 -->
    <el-row justify="center" class="emphasis" v-for="emphasis in emphases">
        {{ emphasis }}
    </el-row>

    <!-- 提供引导按钮 -->
    <el-row justify="center" style="margin-bottom: 20px;">
      <el-col :span="20">
        <el-row justify="center">
          <a :href=button.link v-for="button in buttons" :key="button.name">
            <el-button class="guidance-button" size="default" :disabled="button.disabled" round>
              <el-icon :size="18">
                <component :is="button.component" />
              </el-icon>
              <span class="btn-text">{{ button.name }}</span>
            </el-button>
          </a>
        </el-row>
      </el-col>
    </el-row>

  </div>
</template>

<style scoped>

/* Hero 背景：顶部柔和品牌色光晕 */
.hero {
  position: relative;
}

.hero::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 560px;
  z-index: -1;
  background:
    radial-gradient(640px 300px at 12% -4%, rgba(50, 115, 220, 0.10), transparent 70%),
    radial-gradient(640px 300px at 88% -4%, rgba(124, 92, 240, 0.10), transparent 70%);
  pointer-events: none;
}

/* 最新消息胶囊列表 */
.news-list {
  padding-top: 24px;
}

.news-chip {
  display: flex;
  align-items: center;
  margin: 8px 0;
  padding: 9px 16px;
  border-radius: 999px;
  border: 1px solid var(--pcd-border);
  background: #ffffff;
  box-shadow: 0 2px 8px rgba(16, 24, 40, 0.05);
}

.news-chip.highlight {
  border-color: rgba(224, 82, 82, 0.35);
  background: #fff5f5;
}

.news-dot {
  flex-shrink: 0;
  width: 8px;
  height: 8px;
  margin-right: 12px;
  border-radius: 50%;
  background: var(--pcd-accent);
}

.news-chip.highlight .news-dot {
  background: #e05252;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(224, 82, 82, 0.4); }
  50% { box-shadow: 0 0 0 5px rgba(224, 82, 82, 0); }
}

.news-text {
  font-family: 'Inter', 'Noto Sans', sans-serif;
  font-size: 14.5px;
  line-height: 1.5;
  color: var(--pcd-text);
  text-align: left;
}

/* 文章标题字体、字间距、居中排布、字号 */
.paper-title {
  font-family: 'Google Sans', sans-serif;
  font-weight: 600;
  letter-spacing: -0.5px;
  line-height: 1.2;
  font-size: 3rem;
  margin: 36px;
  text-align: center;
}

/* 标题补充：渐变文字 */
.title-supp {
  background: linear-gradient(120deg, var(--pcd-accent), #7c5cf0);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

/* 姓名和地址按钮 */
.title-button {
  margin: 0px 0px;
  padding: 5px;
  transition: color 0.2s ease;
}

/* 姓名和地址按钮光标悬浮 */
.title-button:hover {
  margin: 0px 0px;
  color: var(--pcd-accent);
}

/* 引导材料按钮：品牌渐变胶囊 */
.guidance-button {
  margin: 8px 5px;
  padding: 18px 15px;
  border: none;
  background: linear-gradient(120deg, var(--pcd-accent), #7c5cf0);
  color: #ffffff;
  transition: transform 0.25s ease, box-shadow 0.25s ease, filter 0.25s ease;
}

.guidance-button:hover {
  background: linear-gradient(120deg, #2b64c4, #6a4de0);
  color: #ffffff;
  transform: translateY(-2px);
  box-shadow: 0 10px 24px rgba(50, 115, 220, 0.35);
}

.guidance-button:active {
  transform: translateY(0);
  filter: brightness(0.96);
}

/* 姓名属性 */
.author {
  font-family: 'Google Sans', sans-serif;
  font-size: 20px;
}

/* 姓名上标属性 */
.name_sup {
  font-family: 'Google Sans', sans-serif;
  color: #606266;
}

/* 地址属性 */
.address {
  font-family: 'Google Sans', sans-serif;
  font-size:  20px;
}

/* 地址上标属性 */
.address_sup {
  font-family: 'Google Sans', sans-serif;
  color: #606266;
}

/* 头像属性 */
.el-avatar {
  margin-right: 5px;
}

/* 共一和通讯文字属性 */
.con-cor {
  font-family: 'Google Sans', sans-serif;
  font-size: 16px;
  margin: 18px 0px;
  text-align: center;
  white-space: pre;
  color: var(--pcd-muted);
}

/* 强调信息属性 */
.emphasis {
  font-family: 'Google Sans', sans-serif;
  color: #d97706;
  font-weight: bold;
  margin: 8px;
  font-size: 22px;
  text-align: center;
}

/* 引导材料按钮文字属性 */
.btn-text {
  font-family: 'Google Sans', sans-serif;
  font-size: 16px;
  color: #ffffff;
}

.logo {
  width: 150px;
  height: 150px;
  border-radius: 50%;
  box-shadow: #ced3dc 0px 0px 3px 2px;
  margin-top: 40px;
}

/* 手机端样式处理 */
a:-webkit-any-link {
  text-decoration: none;
}

/* 取消鼠标焦点悬浮在链接上的颜色装饰 */
a:hover {
  color: inherit;
  border-bottom: none;
}

/* 链接装饰，取消下划线和链接颜色 */
a {
	text-decoration: None;
	color: inherit;
}

@media (max-width: 600px) {
  .paper-title {
    font-size: 2.1rem;
    margin: 24px;
  }

  .news-chip {
    border-radius: 16px;
  }

  .emphasis {
    font-size: 18px;
  }
}

</style>