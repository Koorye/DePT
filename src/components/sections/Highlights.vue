<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'

// 核心成果数据（与论文摘要中的数字保持一致）
const stats = [
  {
    value: '6',
    label: 'PT Baselines Enhanced',
    sub: 'CoOp · CoCoOp · KgCoOp · MaPLe · PromptSRC',
    accent: true,
  },
  {
    value: '+2.05%',
    label: 'New-Task Accuracy',
    sub: 'up to, vs. baseline',
    accent: false,
  },
  {
    value: '+1.45%',
    label: 'Harmonic Mean',
    sub: 'on the CoOp baseline',
    accent: false,
  },
  {
    value: '11',
    label: 'Datasets',
    sub: 'image classification benchmarks',
    accent: false,
  },
]

// 跨模型适配性：轮播展示不同骨干策略的仿真增益
const policyGains = [
  { policy: 'OpenVLA', value: '+50.6%' },
  { policy: 'Octo', value: '+29.7%' },
  { policy: 'π₀', value: '+8.9%' },
]

const gainIndex = ref(0)
let gainTimer = null

onMounted(() => {
  gainTimer = setInterval(() => {
    gainIndex.value = (gainIndex.value + 1) % policyGains.length
  }, 2600)
})

onBeforeUnmount(() => clearInterval(gainTimer))
</script>

<template>
  <div class="highlights">
    <el-row justify="center">
      <el-col :xs="28" :sm="24" :md="20" :lg="16" :xl="12">
        <div class="stat-grid">
          <div
            class="stat-card"
            v-for="stat in stats"
            :key="stat.label"
            :class="{ accent: stat.accent }"
          >
            <div class="stat-value">{{ stat.value }}</div>
            <div class="stat-label">{{ stat.label }}</div>
            <div class="stat-sub">{{ stat.sub }}</div>
          </div>

          <!-- 跨模型适配性：不同策略增益轮播卡 -->
          <div class="stat-card cycler">
            <Transition name="gain" mode="out-in">
              <div class="stat-value" :key="gainIndex">{{ policyGains[gainIndex].value }}</div>
            </Transition>
            <div class="stat-label">Simulation Gain</div>
            <Transition name="gain" mode="out-in">
              <div class="stat-sub" :key="gainIndex">{{ policyGains[gainIndex].policy }} · vs. baseline</div>
            </Transition>
          </div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>

.highlights {
  margin: 28px 0 8px;
}

/* 数据卡片 */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
}

.stat-card {
  background: #ffffff;
  border: 1px solid var(--pcd-border);
  border-radius: 16px;
  padding: 22px 16px;
  text-align: center;
  box-shadow: 0 2px 10px rgba(16, 24, 40, 0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--pcd-shadow-lift);
  border-color: rgba(50, 115, 220, 0.35);
}

.stat-card.accent {
  background: linear-gradient(150deg, #3273dc, #7c5cf0);
  border: none;
}

.stat-card.accent:hover {
  box-shadow: 0 12px 32px rgba(50, 115, 220, 0.4);
}

.stat-value {
  font-family: 'Google Sans', sans-serif;
  font-size: 34px;
  font-weight: 700;
  line-height: 1.2;
  color: var(--pcd-accent);
}

.stat-card.accent .stat-value,
.stat-card.cycler .stat-value {
  color: #ffffff;
}

/* 轮播卡：渐变底色突出跨模型适配 */
.stat-card.cycler {
  background: linear-gradient(150deg, #7c5cf0, #3273dc);
  border: none;
}

.stat-card.cycler:hover {
  box-shadow: 0 12px 32px rgba(124, 92, 240, 0.4);
}

.stat-label {
  font-family: 'Google Sans', sans-serif;
  font-size: 15px;
  font-weight: 600;
  color: var(--pcd-heading);
  margin-top: 6px;
}

.stat-card.accent .stat-label,
.stat-card.cycler .stat-label {
  color: #ffffff;
}

.stat-sub {
  font-family: 'Inter', 'Noto Sans', sans-serif;
  font-size: 12.5px;
  color: var(--pcd-muted);
  margin-top: 4px;
}

.stat-card.accent .stat-sub,
.stat-card.cycler .stat-sub {
  color: rgba(255, 255, 255, 0.85);
}

/* 轮播切换动画 */
.gain-enter-active,
.gain-leave-active {
  transition: opacity 0.25s ease, transform 0.25s ease;
}

.gain-enter-from {
  opacity: 0;
  transform: translateY(6px);
}

.gain-leave-to {
  opacity: 0;
  transform: translateY(-6px);
}

@media (max-width: 992px) {
  .stat-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 480px) {
  .stat-value {
    font-size: 28px;
  }

  .stat-card {
    padding: 16px 10px;
  }
}

</style>
