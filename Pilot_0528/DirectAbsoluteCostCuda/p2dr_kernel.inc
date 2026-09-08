// Projection-only CUDA path for DirectAbsoluteCost2.
// This keeps depth/angle priors in Python and only accelerates projection
// residual/Jacobian/Hessian assembly.

__global__ void fused_p2dr_kernel(
    const float *__restrict__ pose_data_q, // [B, N, 12]
    const float *__restrict__ f_r,         // [B, C, H, W]
    const float *__restrict__ f_q,         // [B, C, H, W]
    const float *__restrict__ cam_data_q,  // [B, N, 6]
    const float *__restrict__ p3D,         // [B, P, 3]
    const float *__restrict__ c_ref,       // [B, 1, H, W]
    const float *__restrict__ c_query,     // [B, 1, H, W]
    const float *__restrict__ p2d_r,       // [B, P, 2]
    const float *__restrict__ visible_r,   // [B, P]
    int B, int N, int P, int C, int H, int W,
    double *__restrict__ grad_out, // [B, N, 6]
    double *__restrict__ Hess_out, // [B, N, 6, 6]
    float *__restrict__ cost_out,  // [B, N, P]
    float *__restrict__ valid_out) // [B, N, P]
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * N * P)
        return;

    int pn = tid % (N * P);
    int n = pn / P;
    int p = pn % P;
    int b = tid / (N * P);

    cost_out[(b * N + n) * P + p] = 0.0f;
    valid_out[(b * N + n) * P + p] = 0.0f;

    const float inv_Wm1 = 2.0f / float(W - 1);
    const float inv_Hm1 = 2.0f / float(H - 1);
    const float eps_x = 2.0f * 4.0f / float(W - 1);
    const float eps_y = 2.0f * 4.0f / float(H - 1);

    const float *pr = p2d_r + ((b * P) + p) * 2;
    float ur = pr[0], vr = pr[1];
    float xnorm_r = ur * inv_Wm1 - 1.0f;
    float ynorm_r = vr * inv_Hm1 - 1.0f;
    bool bound_r = (xnorm_r >= -1.0f + eps_x && xnorm_r <= 1.0f - eps_x &&
                    ynorm_r >= -1.0f + eps_y && ynorm_r <= 1.0f - eps_y);
    bool ref_ok = bound_r && (visible_r[(b * P) + p] > 0.5f);
    if (!ref_ok)
        return;

    float fr_feat[32];
    float fr_norm2 = 0.0f;
    for (int c = 0; c < C; ++c)
    {
        const float *imr = f_r + ((b * C + c) * H) * W;
        float v = sample_by_norm(imr, xnorm_r, ynorm_r, H, W);
        fr_feat[c] = v;
        fr_norm2 += v * v;
    }
    float inv_sqrt_fr = rsqrtf(fr_norm2 + 1e-6f);

    const float *Pw = p3D + ((b * P + p) * 3);
    float Xw = Pw[0], Yw = Pw[1], Zw = Pw[2];
    const float *Pq = pose_data_q + ((b * N + n) * 12);

    float Rq[9], tq[3];
#pragma unroll
    for (int i = 0; i < 9; i++)
        Rq[i] = Pq[i];
#pragma unroll
    for (int i = 0; i < 3; i++)
        tq[i] = Pq[9 + i];

    float Xq = Rq[0] * Xw + Rq[1] * Yw + Rq[2] * Zw + tq[0];
    float Yq = Rq[3] * Xw + Rq[4] * Yw + Rq[5] * Zw + tq[1];
    float Zq = Rq[6] * Xw + Rq[7] * Yw + Rq[8] * Zw + tq[2];
    if (Zq <= 1e-4f)
        return;

    const float *Cq = cam_data_q + ((b * N + n) * 6);
    float fxq = Cq[2], fyq = Cq[3], cxq = Cq[4], cyq = Cq[5];
    float uq = Xq / Zq * fxq + cxq;
    float vq = Yq / Zq * fyq + cyq;
    float xnorm_q = uq * inv_Wm1 - 1.0f;
    float ynorm_q = vq * inv_Hm1 - 1.0f;
    bool bound_q = (xnorm_q >= -1.0f + eps_x && xnorm_q <= 1.0f - eps_x &&
                    ynorm_q >= -1.0f + eps_y && ynorm_q <= 1.0f - eps_y);
    // 匹配 Python project_p3d 的 valid2: 投影点必须落在相机图像内 [0,camW-1]x[0,camH-1]。
    // 注意相机尺寸(camW,camH)可能与特征图(W,H)不同, 否则会把相机外的点误纳入, 污染雅可比。
    float camW = Cq[0], camH = Cq[1];
    bool vis_cam = (uq >= 0.0f && uq <= camW - 1.0f && vq >= 0.0f && vq <= camH - 1.0f);
    float valid_q = (bound_q && vis_cam) ? 1.0f : 0.0f;

    float Jf_x[32], Jf_y[32], res_c[32];
    float dx_norm = inv_Wm1, dy_norm = inv_Hm1;
    int base_q = (b * C) * H * W;
    float cost = 0.0f;
    for (int c = 0; c < C; ++c)
    {
        const float *imq = f_q + base_q + c * H * W;
        float fm = sample_by_norm(imq, xnorm_q - dx_norm, ynorm_q, H, W);
        float fp = sample_by_norm(imq, xnorm_q + dx_norm, ynorm_q, H, W);
        float fv = sample_by_norm(imq, xnorm_q, ynorm_q - dy_norm, H, W);
        float fd = sample_by_norm(imq, xnorm_q, ynorm_q + dy_norm, H, W);
        Jf_x[c] = 0.5f * (fp - fm);
        Jf_y[c] = 0.5f * (fd - fv);

        float fq_feat = sample_by_norm(imq, xnorm_q, ynorm_q, H, W);
        float r = fq_feat - fr_feat[c] * inv_sqrt_fr;
        res_c[c] = r;
        cost += r * r;
    }
    cost_out[(b * N + n) * P + p] = cost;
    valid_out[(b * N + n) * P + p] = valid_q;

    float x_val = cost / (0.1f * 0.1f);
    float g_d1 = 2.0f / (x_val + 2.0f);

    const float *cref_ptr = c_ref + (b * H * W);
    const float *cqry_ptr = c_query + (b * H * W);
    float wq_ap = sample_by_norm(cqry_ptr, xnorm_q, ynorm_q, H, W);
    float wr_ap = sample_by_norm(cref_ptr, xnorm_r, ynorm_r, H, W);
    float weight = g_d1 * wq_ap * wr_ap * valid_q;
    if (weight == 0.0f)
        return;

    float invZ = 1.0f / Zq, Z2 = Zq * Zq;
    float Jp2d[6] = {
        fxq * invZ, 0.0f, -fxq * Xq / Z2,
        0.0f, fyq * invZ, -fyq * Yq / Z2};

    float skew[9] = {0.0f, -Zw, Yw, Zw, 0.0f, -Xw, -Yw, Xw, 0.0f};
    float Jp3d[18];
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            float s = 0.0f;
#pragma unroll
            for (int k = 0; k < 3; k++)
                s += Rq[3 * i + k] * skew[3 * k + j];
            Jp3d[i * 6 + j] = -s;
        }
    }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Jp3d[i * 6 + 3 + j] = Rq[3 * i + j];

    float Jfp2d[32 * 6];
    for (int c = 0; c < C; c++)
    {
        float A0 = Jf_x[c] * Jp2d[0] + Jf_y[c] * Jp2d[3];
        float A1 = Jf_x[c] * Jp2d[1] + Jf_y[c] * Jp2d[4];
        float A2 = Jf_x[c] * Jp2d[2] + Jf_y[c] * Jp2d[5];
#pragma unroll
        for (int k = 0; k < 6; k++)
            Jfp2d[c * 6 + k] = A0 * Jp3d[0 * 6 + k] +
                               A1 * Jp3d[1 * 6 + k] +
                               A2 * Jp3d[2 * 6 + k];
    }

    float grad_i[6];
    for (int k = 0; k < 6; k++)
    {
        float s = 0.0f;
        for (int c = 0; c < C; c++)
            s += Jfp2d[c * 6 + k] * res_c[c];
        grad_i[k] = weight * s;
    }

    float Hess_i[36];
    for (int r = 0; r < 6; r++)
    {
        for (int c2 = 0; c2 < 6; c2++)
        {
            float s = 0.0f;
            for (int c = 0; c < C; c++)
                s += Jfp2d[c * 6 + r] * Jfp2d[c * 6 + c2];
            Hess_i[r * 6 + c2] = weight * s;
        }
    }

    int bg = (b * N + n) * 6;
    for (int k = 0; k < 6; k++)
        atomicAdd(grad_out + bg + k, static_cast<double>(grad_i[k]));
    int bH = ((b * N + n) * 6) * 6;
    for (int i = 0; i < 36; i++)
        atomicAdd(Hess_out + bH + i, static_cast<double>(Hess_i[i]));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
residual_jacobian_p2dr_cuda(
    torch::Tensor pose_data_q,
    torch::Tensor f_r,
    torch::Tensor f_q,
    torch::Tensor cam_data_q,
    torch::Tensor p3D,
    torch::Tensor c_ref,
    torch::Tensor c_query,
    torch::Tensor p2d_r,
    torch::Tensor visible_r)
{
    TORCH_CHECK(pose_data_q.is_cuda() && f_r.is_cuda() && f_q.is_cuda() &&
                    cam_data_q.is_cuda() && p3D.is_cuda() && c_ref.is_cuda() &&
                    c_query.is_cuda() && p2d_r.is_cuda() && visible_r.is_cuda(),
                "residual_jacobian_p2dr_cuda: all inputs must be CUDA");
    TORCH_CHECK(pose_data_q.scalar_type() == torch::kFloat32 &&
                    f_r.scalar_type() == torch::kFloat32 &&
                    f_q.scalar_type() == torch::kFloat32 &&
                    cam_data_q.scalar_type() == torch::kFloat32 &&
                    p3D.scalar_type() == torch::kFloat32 &&
                    c_ref.scalar_type() == torch::kFloat32 &&
                    c_query.scalar_type() == torch::kFloat32 &&
                    p2d_r.scalar_type() == torch::kFloat32 &&
                    visible_r.scalar_type() == torch::kFloat32,
                "residual_jacobian_p2dr_cuda: all inputs must be float32");

    auto pose_c = pose_data_q.contiguous();
    auto fr_c = f_r.contiguous();
    auto fq_c = f_q.contiguous();
    auto cam_c = cam_data_q.contiguous();
    auto p3d_c = p3D.contiguous();
    auto cref_c = c_ref.contiguous();
    auto cquery_c = c_query.contiguous();
    auto p2dr_c = p2d_r.contiguous();
    auto vis_c = visible_r.contiguous();

    int64_t B = pose_c.size(0);
    int64_t N = pose_c.size(1);
    int64_t C = fq_c.size(1);
    int64_t H = fq_c.size(2);
    int64_t W = fq_c.size(3);
    int64_t P = p3d_c.size(p3d_c.dim() - 2);
    TORCH_CHECK(C <= 32, "residual_jacobian_p2dr_cuda: C must be <= 32");

    auto opts = pose_c.options().dtype(torch::kFloat32);
    auto opts_d = pose_c.options().dtype(torch::kFloat64);
    torch::Tensor grad = torch::zeros({B, N, 6}, opts_d);
    torch::Tensor Hess = torch::zeros({B, N, 6, 6}, opts_d);
    torch::Tensor cost = torch::zeros({B, N, P}, opts);
    torch::Tensor valid = torch::zeros({B, N, P}, opts);

    int64_t total = B * N * P;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    fused_p2dr_kernel<<<blocks, threads>>>(
        pose_c.data_ptr<float>(),
        fr_c.data_ptr<float>(),
        fq_c.data_ptr<float>(),
        cam_c.data_ptr<float>(),
        p3d_c.data_ptr<float>(),
        cref_c.data_ptr<float>(),
        cquery_c.data_ptr<float>(),
        p2dr_c.data_ptr<float>(),
        vis_c.data_ptr<float>(),
        (int)B, (int)N, (int)P, (int)C, (int)H, (int)W,
        grad.data_ptr<double>(),
        Hess.data_ptr<double>(),
        cost.data_ptr<float>(),
        valid.data_ptr<float>());
    CUDA_CHECK_ERRORS();

    return std::make_tuple((-grad).to(torch::kFloat32), Hess.to(torch::kFloat32), cost, valid);
}
