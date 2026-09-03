<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { FullScreen, Close } from '@element-plus/icons-vue'

// 海报地址（public 目录）
const poster = './poster.webp'

// 海报灯箱状态
const lightboxVisible = ref(false)
const posterZoomed = ref(false)

const openPoster = () => {
  lightboxVisible.value = true
  posterZoomed.value = false
  document.body.style.overflow = 'hidden'
}

const closePoster = () => {
  lightboxVisible.value = false
  posterZoomed.value = false
  document.body.style.overflow = ''
}

// ESC 关闭灯箱
const onKeydown = (e) => {
  if (e.key === 'Escape' && lightboxVisible.value) closePoster()
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown)
  document.body.style.overflow = ''
})
</script>

<template>
  <div id="poster" v-animate-onscroll="'animated fadeInUp'">
    <el-row justify="center">
      <h1 class="section-title">Poster</h1>
    </el-row>

    <el-row justify="center">
      <el-col :xs="28" :sm="24" :md="20" :lg="16" :xl="12">
        <div class="poster-frame" @click="openPoster">
          <img :src="poster" alt="DePT Poster" />
          <div class="poster-overlay">
            <span class="poster-zoom-btn">
              <el-icon :size="15"><FullScreen /></el-icon>
              Click to View Fullscreen
            </span>
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- 海报全屏灯箱 -->
    <Teleport to="body">
      <Transition name="lightbox">
        <div v-if="lightboxVisible" class="poster-lightbox" @click.self="closePoster">
          <button class="lightbox-close" aria-label="Close" @click="closePoster">
            <el-icon :size="20"><Close /></el-icon>
          </button>
          <div class="lightbox-body">
            <img
              :src="poster"
              :class="{ 'is-zoomed': posterZoomed }"
              :style="{ cursor: posterZoomed ? 'zoom-out' : 'zoom-in' }"
              alt="DePT Poster"
              @click="posterZoomed = !posterZoomed"
            />
          </div>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>

<style scoped>

/* 海报展示卡片 */
.poster-frame {
  position: relative;
  border-radius: 16px;
  padding: 10px;
  background: linear-gradient(135deg, #f4f8ff 0%, #eaf1fe 100%);
  border: 1px solid #e0eafc;
  box-shadow: 1px 1px 4px 1px #afafaf;
  cursor: zoom-in;
  transition: transform 0.35s ease, box-shadow 0.35s ease;
  overflow: hidden;
  margin: 20px 0;
}

.poster-frame:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 32px rgba(50, 115, 220, 0.3);
}

.poster-frame img {
  display: block;
  width: 100%;
  border-radius: 10px;
}

/* 悬浮遮罩与放大提示 */
.poster-overlay {
  position: absolute;
  inset: 10px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(15, 30, 60, 0);
  opacity: 0;
  transition: opacity 0.3s ease, background 0.3s ease;
}

.poster-frame:hover .poster-overlay {
  opacity: 1;
  background: rgba(15, 30, 60, 0.35);
}

.poster-zoom-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 22px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.92);
  backdrop-filter: blur(6px);
  color: #3273dc;
  font-family: 'Google Sans', sans-serif;
  font-size: 15px;
  font-weight: 600;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.22);
  transform: translateY(6px);
  transition: transform 0.3s ease;
}

.poster-frame:hover .poster-zoom-btn {
  transform: translateY(0);
}

/* 手机端样式处理 */
@media (max-width: 600px) {
  .poster-frame {
    padding: 6px;
    border-radius: 12px;
  }

  .poster-frame img {
    border-radius: 8px;
  }

  .poster-overlay {
    inset: 6px;
    border-radius: 8px;
  }

  .poster-zoom-btn {
    padding: 7px 14px;
    font-size: 13px;
  }
}

</style>

<style>

/* 灯箱挂载在 body 下，需要使用全局样式 */
.poster-lightbox {
  position: fixed;
  inset: 0;
  z-index: 3000;
  background: rgba(10, 16, 28, 0.9);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  display: flex;
}

.poster-lightbox .lightbox-body {
  width: 100%;
  height: 100%;
  overflow: auto;
  padding: 64px 24px 40px;
  box-sizing: border-box;
  display: flex;
}

.poster-lightbox .lightbox-body img {
  margin: auto;
  width: min(92vw, 1500px);
  border-radius: 8px;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.5);
}

.poster-lightbox .lightbox-body img.is-zoomed {
  width: 230vw;
  max-width: none;
}

.poster-lightbox .lightbox-close {
  position: fixed;
  top: 18px;
  right: 22px;
  z-index: 3001;
  width: 44px;
  height: 44px;
  border: none;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.14);
  backdrop-filter: blur(6px);
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background 0.25s ease, transform 0.25s ease;
}

.poster-lightbox .lightbox-close:hover {
  background: rgba(255, 255, 255, 0.28);
  transform: rotate(90deg);
}

/* 灯箱过渡动画 */
.lightbox-enter-active,
.lightbox-leave-active {
  transition: opacity 0.28s ease;
}

.lightbox-enter-active .lightbox-body img,
.lightbox-leave-active .lightbox-body img {
  transition: transform 0.28s ease;
}

.lightbox-enter-from,
.lightbox-leave-to {
  opacity: 0;
}

.lightbox-enter-from .lightbox-body img,
.lightbox-leave-to .lightbox-body img {
  transform: scale(0.95);
}

@media (max-width: 600px) {
  .poster-lightbox .lightbox-body {
    padding: 56px 10px 24px;
  }

  .poster-lightbox .lightbox-body img {
    width: 94vw;
  }

  .poster-lightbox .lightbox-body img.is-zoomed {
    width: 260vw;
  }

  .poster-lightbox .lightbox-close {
    top: 12px;
    right: 14px;
    width: 38px;
    height: 38px;
  }
}

</style>
