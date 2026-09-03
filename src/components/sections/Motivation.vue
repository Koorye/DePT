<script setup>
import motivation from '../mds/motivation.mdx';
</script>

<template>
  <div v-animate-onscroll="'animated fadeInUp'">
    <el-row justify="center">
        <el-col :xs="28" :sm="24" :md="20" :lg="16" :xl="12">
          <h1>Motivation</h1>
            <el-image src="./motivation.png" class="caption-img" fit="cover" />
            <div class="caption-text">
                <motivation />
            </div>
        </el-col>
    </el-row>

  </div>
</template>

<style scoped>

.caption-img {
    width: 100%;
    margin: 20px 0;
    border-radius: 12px;
    border: 1px solid var(--pcd-border);
    box-shadow: var(--pcd-shadow-soft);
}

</style>
