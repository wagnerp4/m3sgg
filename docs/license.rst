License
=======

DLHM VidSGG is released under the MIT License.

MIT License
-----------

Copyright (c) 2025 DLHM Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Third-Party Licenses
--------------------

This project builds upon and includes code from several open source projects:

STTran
~~~~~~

The STTran implementation is based on the original work:

* **Repository**: https://github.com/yrcong/STTran
* **Paper**: "Spatial-Temporal Transformer for Dynamic Scene Graph Generation" (ICCV 2021)
* **License**: MIT License

FasterRCNN
~~~~~~~~~~

The object detection component uses FasterRCNN from:

* **Repository**: https://github.com/jwyang/faster-rcnn.pytorch
* **License**: MIT License

Action Genome Dataset
~~~~~~~~~~~~~~~~~~~~~

When using the Action Genome dataset, please cite:

* **Website**: https://www.actiongenome.org/
* **Paper**: "Action Genome: Actions as Compositions of Spatio-temporal Scene Graphs" (CVPR 2020)
* **License**: Dataset-specific license (see website for details)

Dependencies
~~~~~~~~~~~~

This project uses various open source Python packages. See the full list in ``pyproject.toml``. 
Major dependencies include:

* **PyTorch**: BSD 3-Clause License
* **NumPy**: BSD 3-Clause License
* **OpenCV**: Apache License 2.0
* **Matplotlib**: Matplotlib License (BSD-style)

Attribution
-----------

If you use DLHM VidSGG in your research, please cite:

.. code-block:: bibtex

   @misc{dlhm_vidsgg2025,
     title={DLHM VidSGG: Dynamic Scene Graph Generation for Videos},
     author={DLHM Team},
     year={2025},
     url={https://github.com/your-username/DLHM_VidSGG}
   }

Additionally, please cite the relevant papers for the models and datasets you use:

STTran Citation
~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @inproceedings{cong2021spatial,
     title={Spatial-Temporal Transformer for Dynamic Scene Graph Generation},
     author={Cong, Yuren and Yang, Wentong and Li, Hongwei and Liao, Jie},
     booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
     pages={16372--16382},
     year={2021}
   }

Action Genome Citation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @inproceedings{ji2020action,
     title={Action genome: Actions as compositions of spatio-temporal scene graphs},
     author={Ji, Jingwei and Krishna, Ranjay and Fei-Fei, Li and Niebles, Juan Carlos},
     booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
     pages={10236--10247},
     year={2020}
   }

Contact
-------

For questions about licensing or to request permission for commercial use, please contact:

* **Email**: [contact-email]
* **GitHub**: https://github.com/your-username/DLHM_VidSGG/issues

Disclaimer
----------

The software is provided "as is" without warranty of any kind. The authors and contributors 
are not responsible for any damages or issues that may arise from using this software.

This software is intended for research and educational purposes. Users are responsible for 
ensuring their use complies with all applicable laws and regulations.
