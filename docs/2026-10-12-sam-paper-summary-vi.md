# Sharpness-Aware Minimization - Ghi chú bài báo và định hướng đồ án

## 1. Thông tin bài báo

- **Tiêu đề:** *Sharpness-Aware Minimization for Efficiently Improving Generalization*
- **Tác giả:** Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur
- **Hội nghị:** ICLR 2021
- **Paper:** [OpenReview](https://openreview.net/forum?id=6Tm1mposlrM) | [arXiv HTML](https://arxiv.org/html/2010.01412v3) | [arXiv PDF](https://arxiv.org/pdf/2010.01412)
- **Code chính thức:** [google-research/sam](https://github.com/google-research/sam)

## 2. Tóm tắt trong một câu

Thay vì chỉ tìm bộ trọng số có training loss thấp tại đúng một điểm, Sharpness-Aware Minimization (SAM) tìm bộ trọng số mà training loss vẫn thấp trong cả một lân cận của điểm đó, qua đó hướng mô hình đến vùng nghiệm phẳng hơn và thường cải thiện khả năng generalization.

## 3. Bài toán mà paper giải quyết

Deep neural network hiện đại thường có rất nhiều tham số và có thể đạt training loss gần bằng 0. Tuy nhiên, hai nghiệm có training loss tương đương vẫn có thể có test performance rất khác nhau. Vì vậy, chỉ tối thiểu hóa training loss không đủ để lựa chọn nghiệm generalize tốt.

Paper khai thác giả thuyết quen thuộc về hình học của loss landscape:

- **Sharp minimum:** loss tăng nhanh khi trọng số bị thay đổi một lượng nhỏ.
- **Flat minimum:** loss vẫn thấp trong một vùng lân cận của trọng số hiện tại.
- Nghiệm phẳng được kỳ vọng ít nhạy với nhiễu hoặc perturbation của trọng số và thường có generalization tốt hơn.

Điểm mới của SAM là biến trực giác này thành một objective có thể tối ưu bằng gradient-based optimization ở quy mô lớn.

## 4. Objective của SAM

Với training loss \(L_S(\mathbf{w})\), SAM giải bài toán:

$$
\min_{\mathbf{w}}
\max_{\lVert\boldsymbol{\epsilon}\rVert_p \leq \rho}
L_S(\mathbf{w}+\boldsymbol{\epsilon})
+ \lambda\lVert\mathbf{w}\rVert_2^2.
$$

Ý nghĩa các thành phần:

- \(\mathbf{w}\): trọng số hiện tại của mô hình.
- \(\boldsymbol{\epsilon}\): perturbation tác động lên trọng số.
- \(\rho\): bán kính lân cận cần bảo vệ. Đây là hyperparameter quan trọng nhất của SAM.
- Inner maximization tìm perturbation trong lân cận làm loss tăng mạnh nhất.
- Outer minimization cập nhật \(\mathbf{w}\) để worst-case loss trong lân cận đó giảm xuống.
- Khi \(\rho=0\), objective trở về empirical risk minimization thông thường.

Paper xuất phát từ một PAC-Bayesian generalization bound trong đó population loss được chặn bởi neighborhood-wise training loss cộng với một thành phần phụ thuộc vào độ lớn trọng số. Tuy nhiên, objective thực tế là một phương pháp được truyền cảm hứng từ bound này, không phải phép tối ưu chính xác toàn bộ bound.

## 5. SAM được tối ưu như thế nào?

Inner maximization chính xác rất đắt. Paper dùng first-order Taylor approximation. Với trường hợp thường dùng \(p=2\), gọi

$$
\mathbf{g}=\nabla_{\mathbf{w}}L_B(\mathbf{w}),
$$

perturbation xấp xỉ là:

$$
\widehat{\boldsymbol{\epsilon}}
= \rho\frac{\mathbf{g}}{\lVert\mathbf{g}\rVert_2}.
$$

Sau đó tính gradient tại điểm bị perturb:

$$
\mathbf{g}_{SAM}
=\nabla_{\mathbf{w}}L_B
(\mathbf{w}+\widehat{\boldsymbol{\epsilon}}),
$$

rồi đưa gradient này cho base optimizer, chẳng hạn SGD, để cập nhật trọng số gốc. Paper bỏ qua second-order term liên quan đến đạo hàm của \(\widehat{\boldsymbol{\epsilon}}(\mathbf{w})\), nhờ đó không cần tạo Hessian.

```mermaid
flowchart LR
    A[Weights w and one mini-batch B] --> B[First forward and backward]
    B --> C[Compute gradient g]
    C --> D[Build perturbation epsilon with norm rho]
    D --> E[Temporarily use w plus epsilon]
    E --> F[Second forward and backward on the same batch]
    F --> G[Restore original weights w]
    G --> H[Base optimizer update using SAM gradient]
```

Điểm cốt lõi là một SAM update có hai lần forward/backward:

1. Tính gradient tại \(\mathbf{w}\) để tạo perturbation.
2. Tính gradient tại \(\mathbf{w}+\widehat{\boldsymbol{\epsilon}}\) để cập nhật \(\mathbf{w}\).

Vì vậy, SAM là một wrapper quanh base optimizer hơn là một optimizer hoàn toàn tách biệt. Có thể dùng SAM với SGD, Adam hoặc optimizer khác, nhưng paper gốc chủ yếu đánh giá với SGD.

## 6. Đóng góp chính

1. **Đề xuất objective min-max có ý nghĩa hình học rõ ràng:** tối thiểu hóa cả loss value và local sharpness.
2. **Đưa ra thuật toán first-order thực dụng:** chỉ cần hai lần tính gradient mỗi update, không cần materialize Hessian.
3. **Thực nghiệm rộng vào thời điểm công bố:** CIFAR-10, CIFAR-100, SVHN, Fashion-MNIST, ImageNet và nhiều bài toán finetuning.
4. **Khả năng kết hợp với kỹ thuật khác:** SAM vẫn cải thiện kết quả khi mô hình đã dùng AutoAugment, Cutout, Shake-Shake hoặc ShakeDrop.
5. **Robustness với synthetic label noise:** SAM vượt SGD rõ rệt khi nhãn CIFAR-10 bị thay ngẫu nhiên.
6. **Đề xuất \(m\)-sharpness:** sharpness phụ thuộc vào số mẫu \(m\) được dùng để tính từng perturbation. Trong thí nghiệm của paper, \(m\) nhỏ hơn vừa cho kết quả tốt hơn vừa tương quan tốt hơn với generalization gap.

## 7. Kết quả thực nghiệm đáng nhớ

Các số dưới đây là kết quả do paper báo cáo, không phải kết quả đã chạy lại trong repo này.

| Thiết lập | Không dùng SAM | Dùng SAM | Diễn giải |
|---|---:|---:|---|
| CIFAR-10, WRN-28-10, AutoAugment, 1800 epochs | 2.2% test error | 1.6% | SAM giảm test error dù baseline đã dùng augmentation mạnh |
| CIFAR-100, WRN-28-10, AutoAugment, 1800 epochs | 16.1% | 12.8% | Mức cải thiện lớn hơn trên bài toán khó hơn |
| ImageNet, ResNet-152, 200 epochs | 20.3% top-1 error | 18.5% | So sánh tại cùng số epoch, nhưng mỗi SAM epoch vẫn tốn nhiều compute hơn |
| CIFAR-10 với 80% synthetic label noise | 26.2% accuracy với SGD | 77.9% với SAM | Cho thấy robustness mạnh trong thiết lập nhiễu tổng hợp này |

Paper chạy năm replica độc lập cho các experimental condition chính và báo cáo mean cùng 95% confidence interval. Với phân tích Hessian trên một WideResNet, paper báo cáo maximum Hessian eigenvalue tại convergence xấp xỉ 24 khi dùng SGD và 1.0 khi dùng SAM, phù hợp với việc SAM tìm vùng có curvature thấp hơn.

## 8. Điểm mạnh của paper

- Ý tưởng đơn giản, có thể giải thích trực quan và chuyển thành code tương đối ngắn.
- Objective, approximation và update rule liên kết chặt chẽ với nhau.
- Thực nghiệm bao phủ nhiều architecture, dataset và training regime.
- Baseline được xử lý khá cẩn thận: vì SAM cần hai backward pass, non-SAM baseline được phép chạy cả số epoch chuẩn và số epoch gấp đôi; paper báo cáo kết quả tốt hơn trong hai lần chạy này.
- Có ablation cho \(\rho\), \(p\)-norm, second-order term và số bước của inner maximization.
- Có code chính thức và một báo cáo tái lập độc lập đã tái hiện xu hướng SAM tốt hơn SGD trên phạm vi nhỏ hơn.

## 9. Những điều phải hiểu đúng

### 9.1 “Efficiently” không có nghĩa là rẻ hơn SGD

SAM hiệu quả so với việc giải inner maximization chính xác hoặc thao tác trực tiếp với Hessian. Một SAM update vẫn cần hai forward/backward pass, nên chi phí compute thường gần gấp đôi một update của base optimizer. Cần báo cáo wall-clock time hoặc GPU-hours, không chỉ số epoch.

### 9.2 Flatness không phải đại lượng tuyệt đối

Sharpness đo trong weight space có thể thay đổi khi reparameterize hoặc rescale mạng dù hàm mà mạng biểu diễn không đổi. Dinh et al. (2017) đã chỉ ra có thể tạo các parameterization tương đương nhưng có sharpness rất khác nhau. Vì vậy, không nên diễn giải “flat minimum là nguyên nhân phổ quát của generalization” như một định luật đã được chứng minh. Kết luận an toàn hơn là SAM tối ưu một dạng local weight robustness và đã cho kết quả thực nghiệm tốt trong các thiết lập được đánh giá.

### 9.3 Generalization bound chỉ tạo động lực cho objective

Paper thay thành phần \(h(\lVert\mathbf{w}\rVert_2^2/\rho^2)\) trong bound bằng L2 regularization thực dụng. Inner problem cũng được linearize và second-order term bị bỏ. Do đó có khoảng cách giữa theorem, objective lý tưởng và thuật toán thực thi.

### 9.4 \(\rho=0.05\) chỉ là default ban đầu

Paper tune \(\rho\) trên tập validation với grid
\(\{0.01,0.02,0.05,0.1,0.2,0.5\}\) và nhận thấy 0.05 thường hoạt động tốt. Giá trị phù hợp vẫn phụ thuộc architecture, optimizer, learning rate, weight decay, batch size và parameterization. Trong thí nghiệm label noise, paper dùng \(\rho=0.1\) cho phần lớn mức nhiễu nhưng đổi sang 0.05 ở mức nhiễu 80% để ổn định hơn.

### 9.5 Kết quả label noise có phạm vi hẹp

Paper dùng synthetic symmetric label noise trên CIFAR-10 và giữ validation/test set sạch. Đây không tương đương hoàn toàn với real-world label noise có cấu trúc hoặc instance-dependent noise. Không nên khái quát trực tiếp rằng SAM giải quyết mọi bài toán noisy labels.

### 9.6 \(m\)-sharpness làm distributed implementation thay đổi thuật toán

Tính perturbation riêng trên từng device rồi average gradient không tương đương với tính một perturbation toàn cục trên full batch. Paper xem đây là một đặc tính có lợi, nhưng nhóm phải cố định và mô tả rõ effective perturbation batch size \(m\), số GPU và cách đồng bộ.

## 10. Lưu ý khi implement

- Dùng **cùng một mini-batch** cho cả hai forward/backward pass của một SAM step.
- Sau bước ascent, phải restore đúng trọng số gốc trước khi base optimizer update.
- Chuẩn hóa gradient với một epsilon số học nhỏ để tránh chia cho 0.
- Không áp perturbation cho parameter không có gradient. Cần ghi rõ có loại trừ bias hoặc BatchNorm parameter hay không.
- Xóa hoặc quản lý gradient đúng giữa hai backward pass, tránh cộng dồn ngoài ý muốn.
- **BatchNorm là bẫy quan trọng:** nếu running statistics được cập nhật ở cả hai forward pass thì statistics bị đếm hai lần. Báo cáo tái lập độc lập xử lý bằng cách không cập nhật BatchNorm statistics ở pass thứ hai.
- Nếu dùng dropout hoặc stochastic layers, cần mô tả rõ cách xử lý randomness giữa hai pass.
- Weight decay phải được áp dụng nhất quán qua base optimizer. Không nên vô tình cộng weight decay hai lần.
- AMP/mixed precision cần unscale gradient trước khi tính gradient norm và perturbation.
- Lưu checkpoint theo validation metric, không chọn kết quả dựa trên test set.

## 11. Đề xuất phạm vi đồ án CS2229

Một phạm vi vừa sức nhưng vẫn có giá trị là **tái lập có kiểm soát**, không cố chạy lại toàn bộ paper:

### Câu hỏi chính

> Với cùng architecture và data pipeline trên CIFAR-10, SAM có cải thiện test accuracy và generalization gap so với SGD khi so sánh công bằng theo cả số update lẫn compute budget hay không?

### Thiết lập tối thiểu

- Dataset: CIFAR-10.
- Model: ResNet-18 hoặc WideResNet nhỏ, tùy GPU.
- Baselines: SGD và SAM-SGD.
- Seeds: ít nhất 3, tốt hơn là 5.
- Metrics: test accuracy, training accuracy/loss, generalization gap, thời gian mỗi epoch, tổng wall-clock time và peak GPU memory.
- Ablation chính: \(\rho \in \{0, 0.01, 0.05, 0.1, 0.2\}\).
- Giữ cố định augmentation, batch size, learning-rate schedule, momentum và weight decay giữa các cấu hình, trừ biến đang ablate.

### Hai cách so sánh nên báo cáo song song

1. **Equal epochs/updates:** cho biết lợi ích nếu chấp nhận SAM tốn thêm compute.
2. **Equal compute/wall-clock budget:** cho biết SAM có còn lợi hơn khi baseline được dùng lượng compute tương đương hay không.

### Extension nếu còn thời gian

- Thêm synthetic label noise ở một hoặc hai mức, chẳng hạn 40% và 80%.
- Đo một sharpness proxy hoặc maximum Hessian eigenvalue xấp xỉ và kiểm tra tương quan với generalization gap.
- So sánh SAM-SGD với SAM-Adam, nhưng chỉ nên làm sau khi baseline chính đã ổn định.

## 12. Checklist để tránh kết luận quá mức

- [ ] Không gọi một run tốt nhất là bằng chứng, phải báo cáo mean và dispersion qua nhiều seed.
- [ ] Không tune \(\rho\) trên test set.
- [ ] Không so sánh SAM đã tune kỹ với SGD dùng hyperparameter mặc định yếu.
- [ ] Không chỉ báo cáo accuracy mà bỏ qua compute cost.
- [ ] Không dùng “SAM luôn tốt hơn” nếu chỉ thử một dataset hoặc một model.
- [ ] Không coi training loss cao hơn là thất bại nếu test performance tốt hơn, vì SAM đang tối ưu neighborhood loss.
- [ ] Không đồng nhất robustness với label noise và adversarial robustness. Đây là hai dạng robustness khác nhau.
- [ ] Phân biệt rõ kết quả do paper báo cáo, kết quả tái lập từ nguồn khác và kết quả nhóm tự chạy.
- [ ] Ghi đủ seed, package version, hardware, precision mode và toàn bộ hyperparameter.

## 13. Cấu trúc trình bày đề xuất

1. Vấn đề: training loss thấp chưa đảm bảo generalization.
2. Trực giác: flat neighborhood tốt hơn một low-loss point sắc nhọn.
3. Objective min-max của SAM.
4. Xấp xỉ first-order và hai gradient pass.
5. Kết quả chính của paper.
6. Thiết kế tái lập của nhóm và nguyên tắc fair comparison.
7. Kết quả, ablation theo \(\rho\), compute overhead.
8. Hạn chế và kết luận có phạm vi.

## 14. Tài liệu tham khảo

- Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021). *Sharpness-aware minimization for efficiently improving generalization*. International Conference on Learning Representations. https://openreview.net/forum?id=6Tm1mposlrM
- Dinh, L., Pascanu, R., Bengio, S., & Bengio, Y. (2017). Sharp minima can generalize for deep nets. *Proceedings of the 34th International Conference on Machine Learning*, 70, 1019-1028. https://proceedings.mlr.press/v70/dinh17b.html
- Lin, Y. (2021). *[Re] Sharpness-Aware Minimization for Efficiently Improving Generalization*. Independent reproduction report. https://yulonglin.com/papers/sam_replication.pdf
- Google Research. (n.d.). *SAM: Sharpness-Aware Minimization for Efficiently Improving Generalization* [Source code]. GitHub. https://github.com/google-research/sam

## 15. Phạm vi kiểm tra của note

Note này được đối chiếu với full text arXiv v3 của paper, repository chính thức, một báo cáo tái lập độc lập và paper của Dinh et al. về hạn chế của sharpness trong weight space. Các con số trong Mục 7 là số do Foret et al. báo cáo và chưa phải kết quả tái lập của nhóm.

> **AI disclosure:** Note được soạn với sự hỗ trợ của công cụ AI cho việc tra cứu, đối chiếu nguồn và tổng hợp. Nhóm cần kiểm tra lại code, hyperparameter và kết quả thực nghiệm trước khi đưa vào báo cáo chính thức.
