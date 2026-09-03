<script>

import TopNav from './components/TopNav.vue'
import Main from './components/Main.vue'

export default {
  components: {
    TopNav,
    Main,
  }
}

</script>

<template>
    <div>
      
      <TopNav/>
      <el-container>
        <el-main class="page-main"><Main/></el-main>
      </el-container>
      
      <!-- 回到顶部 -->
      <el-backtop :right="40" :bottom="80" />
    </div>
</template>

<style>

/* ===== 设计变量 ===== */
:root {
  --pcd-accent: #3273dc;
  --pcd-accent-strong: #1d5fc2;
  --pcd-heading: #101828;
  --pcd-text: #344054;
  --pcd-muted: #667085;
  --pcd-border: #e7ecf3;
  --pcd-surface: #f8fafd;
  --pcd-shadow-soft: 0 4px 20px rgba(16, 24, 40, 0.08);
  --pcd-shadow-lift: 0 12px 32px rgba(16, 24, 40, 0.14);
}

html {
  background: #ffffff;
  scroll-behavior: smooth;
}

/* 锚点定位时预留固定导航栏高度 */
[id] {
  scroll-margin-top: 76px;
}

body {
  background: #ffffff;
  color: var(--pcd-text);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

::selection {
  background: rgba(50, 115, 220, 0.18);
}

h1 {
  font-family: 'Google Sans', sans-serif;
  letter-spacing: 0px;
  font-size: 28px;
  margin: 0px;
  text-align: center;
}

/* ===== 统一的章节标题系统（论文主标题除外） ===== */
.page-main h1:not(.paper-title) {
  font-size: 30px;
  font-weight: 700;
  letter-spacing: 0.2px;
  color: var(--pcd-heading);
  margin: 56px 0 28px;
  padding-bottom: 16px;
  position: relative;
  text-align: center;
}

.page-main h1:not(.paper-title)::after {
  content: "";
  position: absolute;
  left: 50%;
  bottom: 0;
  transform: translateX(-50%);
  width: 56px;
  height: 3px;
  border-radius: 3px;
  background: linear-gradient(90deg, var(--pcd-accent), #7aa5f5);
}

h2, h3, h4, h5, h6 {
  font-family: 'Google Sans', sans-serif;
  letter-spacing: 0px;
  color: var(--pcd-heading);
}

.el-header {
  padding: 0px;
}

/* 顶部导航栏高度，避免内容被遮挡 */
.el-main.page-main {
  padding-top: 60px;
}

.el-footer {
  padding: 0px;
}

div {
  font-family: 'Inter', 'Noto Sans', sans-serif;
}

p, li {
  font-family: 'Inter', 'Noto Sans', sans-serif;
  font-size: 16px;
  line-height: 1.8rem;
  text-align: justify;
  text-align-last: start;
  word-spacing: normal;
  hyphens: auto;
  color: var(--pcd-text);
}

code {
  font-family: 'Consolas', monospace;
  border-radius: 6px;
  padding: 2px 6px;
  background: #f2f5f9;
  border: 1px solid var(--pcd-border);
  color: #1f2937;
  font-size: 90%;
}

pre {
  border-radius: 10px;
  padding: 14px 18px;
  background: #f8fafc;
  border: 1px solid var(--pcd-border);
  line-height: 1.5rem;
}

pre:not(pre[id=bibtex]) {
  overflow-x: auto;
}

pre code {
  font-size: 16px;
  color: #111827;
  border: none;
  padding: 0px;
  background: none;
}

blockquote {
  border-left: 4px solid var(--pcd-accent);
  padding: 12px 20px;
  margin: 0px;
  background: var(--pcd-surface);
  border-radius: 0 10px 10px 0;
}

blockquote > p {
  margin-block-start: 0.5em;
  margin-block-end: 0.5em;
}

/* 链接颜色装饰 */
a {
  color: var(--pcd-accent);
  text-decoration: none;
  transition: color 0.2s ease;
}

/* 鼠标焦点悬浮在链接上的颜色装饰 */
a:hover {
  color: var(--pcd-accent-strong);
  border-bottom: none;
}

p > img {
  width: 100%;
  display: block;
  margin: 16px auto;
  border-radius: 12px;
  border: 1px solid var(--pcd-border);
  box-shadow: var(--pcd-shadow-soft);
}

table {
  border-collapse: collapse;
  width: max-content;
  max-width: 100%;
  margin: 0 auto;
  display: block;
  overflow-x: auto;
}

thead {
  border-bottom-width: 1px;
  border-top-width: 2px;
  border-left-width: 0px;
  border-right-width: 0px;
  border-style: solid;
  border-color: rgb(0 0 0);
}

tbody tr:last-child {
  border-bottom-width: 2px;
  border-top-width: 0px;
  border-left-width: 0px;
  border-right-width: 0px;
  border-style: solid;
  border-color: rgb(0 0 0);
}

th, td {
  padding-left: 1rem;
  padding-right: 1rem;
}

.katex-display {
  overflow-x: auto;
  overflow-y: hidden;
  padding: 4px 0;
}

/* Element Plus 图片圆角跟随容器 */
.el-image {
  border-radius: 12px;
}

.el-image__inner {
  border-radius: inherit;
}

</style>