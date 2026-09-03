<script setup>
import experiment from '../mds/experiment.mdx';
import experiment2 from '../mds/experiment2.mdx';
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
            <el-image src="./exp2.png" class="figure-img" fit="cover" />
            <experiment2 class="figure-text" />
        </el-col>
    </el-row>

  </div>
</template>

<style scoped>

.title {
  margin-bottom: 20px;
}

/* 图 1：与文字左右排版 */
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

/* 后续图：全栏 + 下方配文字 */
.figure-img {
    width: 100%;
    margin: 20px 0 0;
    border-radius: 12px;
    border: 1px solid var(--pcd-border);
    box-shadow: var(--pcd-shadow-soft);
}

.figure-text {
    margin: 16px 0 20px;
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
