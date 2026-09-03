<script setup>
import experiment from '../mds/experiment.mdx';
</script>

<template>
  <div id="experiments" v-animate-onscroll="'animated fadeInUp'">
    <el-row justify="center">
        <el-col :xs="28" :sm="24" :md="20" :lg="16" :xl="12">
            <h1 class="title">Experiment</h1>
            <div class="caption-box">
                <el-image src="./exp1.png" class="caption-img" fit="cover" />
                <experiment class="caption-text" />
            </div>
            <el-image src="./exp2.png" class="caption-img" fit="cover" />
        </el-col>
    </el-row>

  </div>
</template>

<style scoped>

.title {
  margin-bottom: 20px;
}

.caption-box {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 28px;
    margin: 20px 0;
}

.caption-img {
    width: 120%;
    max-width: 460px;
    border-radius: 12px;
    border: 1px solid var(--pcd-border);
    box-shadow: var(--pcd-shadow-soft);
}

.caption-text {
    margin: 0;
}

@media (max-width: 992px) {
  .caption-box {
    flex-direction: column;
    gap: 16px;
  }

  .caption-img {
    width: 100%;
    max-width: 100%;
  }

  .caption-text {
    margin: 0;
  }
}

</style>
