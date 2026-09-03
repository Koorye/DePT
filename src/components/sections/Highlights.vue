<script setup>
// 核心成果数据（与论文保持一致）
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

.stat-card.accent .stat-value {
  color: #ffffff;
}

.stat-label {
  font-family: 'Google Sans', sans-serif;
  font-size: 15px;
  font-weight: 600;
  color: var(--pcd-heading);
  margin-top: 6px;
}

.stat-card.accent .stat-label {
  color: #ffffff;
}

.stat-sub {
  font-family: 'Inter', 'Noto Sans', sans-serif;
  font-size: 12.5px;
  color: var(--pcd-muted);
  margin-top: 4px;
}

.stat-card.accent .stat-sub {
  color: rgba(255, 255, 255, 0.85);
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
