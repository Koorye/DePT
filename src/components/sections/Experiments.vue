<script setup>
import experiment from '../mds/experiment.mdx';
</script>

<template>
  <div id="experiments" v-animate-onscroll="'animated fadeInUp'">
    <el-row justify="center">
        <el-col :xs="28" :sm="24" :md="20" :lg="16" :xl="12">
            <h1 class="title">Experiment</h1>
            <el-image src="./exp1.png" class="caption-img" fit="cover" />
            <experiment class="caption-text" />
            <el-image src="./exp2.png" class="caption-img" fit="cover" />
        </el-col>
    </el-row>

  </div>
</template>

<style scoped>

.title {
  margin-bottom: 20px;
}

.caption-img {
    width: 100%;
    margin: 20px 0 0;
    border-radius: 12px;
    border: 1px solid var(--pcd-border);
    box-shadow: var(--pcd-shadow-soft);
}

.caption-text {
    margin: 20px 0;
}

</style>
