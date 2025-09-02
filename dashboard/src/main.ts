import { mount } from 'svelte'
import './styling/index.css'
import './styling/app.css'
import App from './App.svelte'

const app = mount(App, {
  target: document.getElementById('app')!,
})

export default app
