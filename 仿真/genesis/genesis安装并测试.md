# genesis安装和示例跑通

## genesis安装

```python
pip install genesis-world  # Requires Python>=3.10,<3.14;
```

## 示例跑通

```python
import genesis as gs

gs.init(backend=gs.cpu, logging_level="debug")

scene = gs.Scene(
    show_viewer=True,

)


plane = scene.add_entity(
    gs.morphs.Plane(),
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

scene.build()
for i in range(100):
    scene.step()
```

![2025-09-10_15-35.jpg](https://cdn.jsdelivr.net/gh/zilong-ding/note-gen-image-sync@main/700dce0f-5ccd-4449-976a-0df2390c1f34.jpeg)

### 遇到的问题

scene.build迟迟没有成功类似github上的的[issue](https://github.com/Genesis-Embodied-AI/Genesis/issues/1533)，一直卡在

```
[Genesis] [09:36:53] [DEBUG] Trying to create OpenGL Context for PYOPENGL_PLATFORM='native'...
```

这一步。

### 解决方法

设置英伟达显卡为首选

```python
# import os
# os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
# os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
```

或者是

```bash
sudo prime-select nvidia
sudo reboot
```
