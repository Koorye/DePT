<script lang="ts" setup>

import { ref, onMounted, onBeforeUnmount } from 'vue'
import { House, Reading, ArrowDown } from '@element-plus/icons-vue'

// 个人主页地址
const homepage = 'https://koorye.github.io/home/'

// 更多研究地址（主页 Publications 部分）
const more_research = 'https://koorye.github.io/home/#publications'

// 页内锚点导航（与各 section 的 id 对应）
const sections = [
  { id: 'poster', label: 'Poster' },
  { id: 'abstract', label: 'Abstract' },
  { id: 'method', label: 'Method' },
  { id: 'experiments', label: 'Experiments' },
  { id: 'citation', label: 'BibTeX' },
]

// 滚动高亮当前所在章节
const active_section = ref('')

const onScroll = () => {
  let current = ''
  for (const s of sections) {
    const el = document.getElementById(s.id)
    if (el && el.getBoundingClientRect().top <= 100) current = s.id
  }
  active_section.value = current
}

onMounted(() => window.addEventListener('scroll', onScroll, { passive: true }))
onBeforeUnmount(() => window.removeEventListener('scroll', onScroll))

// 其他研究列表，link 优先使用项目主页，没有则使用 GitHub 仓库
interface Paper {
  name: string
  venue: string
  link: string
}

const papers: Paper[] = [
  {
    name: 'RoboCOIN: Open-Sourced Bimanual Robotic Data Collection',
    venue: '2025',
    link: 'https://flagopen.github.io/RoboCOIN/',
  },
  {
    name: 'Policy Contrastive Decoding',
    venue: 'ICLR 2026',
    link: 'https://koorye.github.io/PCD/',
  },
  {
    name: 'InSpire: VLA Models with Intrinsic Spatial Reasoning',
    venue: 'ICRA 2026',
    link: 'https://koorye.github.io/Inspire/',
  },
  {
    name: 'Skip Tuning: Pre-trained VLMs are Effective Adapters Themselves',
    venue: 'CVPR 2025',
    link: 'https://koorye.github.io/SkipTuning/',
  },
  {
    name: 'A Closer Look at Conditional Prompt Tuning for VLMs',
    venue: 'IJCV 2026',
    link: 'https://github.com/Koorye/CaPT/',
  },
  {
    name: 'Decoupled Prompt Tuning',
    venue: 'CVPR 2024',
    link: 'https://koorye.github.io/DePT/',
  },
]

// 点击论文跳转对应项目主页
const gotoPaper = (paper: Paper) => {
  window.location.href = paper.link
}

// 下拉栏展开状态，控制箭头方向
const research_open = ref(false)

</script>

<template>
  <nav class="top-nav">
    <a href="#top" class="brand">DePT</a>

    <div class="nav-links">
      <a
        v-for="s in sections"
        :key="s.id"
        :href="`#${s.id}`"
        class="nav-anchor"
        :class="{ active: active_section === s.id }"
      >{{ s.label }}</a>
    </div>

    <div class="nav-side">
      <a :href="homepage" class="nav-link">
        <el-icon :size="15"><House /></el-icon>
        <span class="nav-text">Homepage</span>
      </a>

      <el-dropdown
        class="nav-dropdown"
        trigger="hover"
        popper-class="research-popper"
        @command="gotoPaper"
        @visible-change="(visible: boolean) => research_open = visible"
      >
        <span class="nav-link">
          <el-icon :size="15"><Reading /></el-icon>
          <span class="nav-text">More Research</span>
          <el-icon :size="12" class="nav-arrow" :class="{ open: research_open }"><ArrowDown /></el-icon>
        </span>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item v-for="paper in papers" :key="paper.name" :command="paper">
              <div class="paper-item">
                <span class="paper-name">{{ paper.name }}</span>
                <span class="paper-venue">{{ paper.venue }}</span>
              </div>
            </el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </div>
  </nav>
</template>

<style scoped>

.top-nav {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 2000;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 28px;
  background: rgba(255, 255, 255, 0.82);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--pcd-border);
}

/* 品牌标识 */
.brand {
  font-family: 'Google Sans', sans-serif;
  font-weight: 700;
  font-size: 17px;
  letter-spacing: 0.5px;
  background: linear-gradient(120deg, var(--pcd-accent), #7c5cf0);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
}

.brand:hover {
  opacity: 0.8;
}

/* 页内锚点导航 */
.nav-links {
  display: flex;
  align-items: center;
  gap: 4px;
}

.nav-anchor {
  padding: 6px 12px;
  border-radius: 999px;
  color: var(--pcd-muted);
  font-family: 'Google Sans', sans-serif;
  font-size: 14px;
  line-height: 20px;
  white-space: nowrap;
  transition: color 0.2s ease, background 0.2s ease;
}

.nav-anchor:hover {
  color: var(--pcd-accent);
  background: rgba(50, 115, 220, 0.08);
}

.nav-anchor.active {
  color: var(--pcd-accent);
  background: rgba(50, 115, 220, 0.1);
  font-weight: 600;
}

/* 右侧功能链接 */
.nav-side {
  display: flex;
  align-items: center;
}

.nav-link {
  display: inline-flex;
  align-items: center;
  margin-left: 20px;
  color: #4a4a4a;
  font-family: 'Google Sans', sans-serif;
  font-size: 15px;
  line-height: 24px;
  white-space: nowrap;
  cursor: pointer;
  transition: color 0.2s;
}

.nav-link:hover {
  color: var(--pcd-accent);
  border-bottom: none;
}

.nav-text {
  margin-left: 5px;
}

/* 下拉栏触发器外层包裹 */
.nav-dropdown {
  vertical-align: middle;
}

/* 下拉箭头展开旋转 */
.nav-arrow {
  margin-left: 4px;
  transition: transform 0.2s;
}

.nav-arrow.open {
  transform: rotate(180deg);
}

/* 手机端样式处理：隐藏页内锚点，保留品牌与功能入口 */
@media (max-width: 1100px) {
  .nav-links {
    display: none;
  }
}

@media (max-width: 600px) {
  .top-nav {
    padding: 8px 16px;
  }

  .nav-link {
    margin-left: 14px;
    font-size: 14px;
  }
}

</style>

<style>

/* 下拉栏挂载在 body 下，需要使用全局样式 */
.research-popper {
  max-width: min(480px, 86vw);
}

.research-popper .paper-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  min-width: 280px;
}

.research-popper .paper-name {
  font-family: 'Google Sans', sans-serif;
  font-size: 14px;
  line-height: 1.4;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.research-popper .paper-venue {
  flex-shrink: 0;
  font-family: 'Google Sans', sans-serif;
  font-size: 12px;
  color: #3273dc;
  background: #edf3fc;
  border-radius: 10px;
  padding: 1px 8px;
}

/* 菜单项聚焦时同样去掉默认黑框（已有高亮背景作为焦点指示） */
.research-popper .el-dropdown-menu__item:focus,
.research-popper .el-dropdown-menu__item:focus-visible {
  outline: none;
}

</style>
