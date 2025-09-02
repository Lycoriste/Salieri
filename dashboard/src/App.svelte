<!-- src/App.svelte -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Plotly from 'plotly.js-dist-min';

  const PLOT_DIV = 'plotly-div';
  let intervalId: number;

  async function updatePlot() {
    try {
      const response = await fetch('http://localhost:7777/graph');
      if (!response.ok) {
        console.error("Failed to fetch plot data:", response.statusText);
        return;
      }
      
      const plotJsonString = await response.json();
      const plotData = JSON.parse(plotJsonString);

      Plotly.react(PLOT_DIV, plotData.data, plotData.layout);

    } catch (error) {
      console.error("Error updating plot:", error);
    }
  }

  onMount(() => {
    updatePlot();
    intervalId = setInterval(updatePlot, 5000);
  });

  onDestroy(() => {
    clearInterval(intervalId);
  });
</script>

<main>
  <h1 class="title">Dashboard</h1>
  <div class="flex w-full h-96 justify-center mx-4" id={PLOT_DIV}></div>
</main>