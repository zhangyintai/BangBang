classdef PEPS_test_6 < matlab.mixin.Copyable
    properties
        a = 0;
        b = 0;
        d = 2;
        D = 4;
        chimax = 16;
        CA = 0;
        TA = 0;
        CB = 0;
        TB = 0;
    end
    
    methods

        function SetValues(obj, aup, adown, bup, bdown, d, D, chimax)
            obj.a = zeros(1, 1, 1, 1, d) + 0j;
            obj.b = zeros(1, 1, 1, 1, d) + 0j;
            obj.a(1, 1, 1, 1, 1) = aup;
            obj.a(1, 1, 1, 1, 2) = adown;
            obj.b(1, 1, 1, 1, 1) = bup;
            obj.b(1, 1, 1, 1, 2) = bdown;
            obj.d = d;
            obj.D = D;
            obj.chimax = chimax;
        end

        function Cubize(obj)
            size_a = size(obj.a);
            size_b = size(obj.b);
            max_size = max([size_a(1:4), size_b(1:4)]);
            temp_a = zeros(max_size, max_size, max_size, max_size, obj.d);
            temp_b = zeros(max_size, max_size, max_size, max_size, obj.d);
            temp_a(1:size(obj.a, 1), 1:size(obj.a, 2), 1:size(obj.a, 3), 1:size(obj.a, 4), 1:obj.d) = obj.a;
            temp_b(1:size(obj.b, 1), 1:size(obj.b, 2), 1:size(obj.b, 3), 1:size(obj.b, 4), 1:obj.d) = obj.b;
            obj.a = temp_a;
            obj.b = temp_b;
        end

        function CTMInitialize(obj, obj_low, form)

            if form == "ones"
                f = @(x)ones(x);
            elseif form == "rand"
                f = @(x)rand(x);
            elseif form == "eye"
                f = @(x)eye(x);
            end

            obj.CA = {f([1, 1]), ...
                      f([1, 1]), ...
                      f([1, 1]), ...
                      f([1, 1]),};
            
            obj.CB = {f([1, 1]), ...
                      f([1, 1]), ...
                      f([1, 1]), ...
                      f([1, 1]),};

            obj.TA = {RESHAPE(f([size(obj.a, 2), size(obj_low.a, 2)]), [1, size(obj.a, 2), size(obj_low.a, 2), 1]), ...
                      RESHAPE(f([size(obj.a, 3), size(obj_low.a, 3)]), [1, size(obj.a, 3), size(obj_low.a, 3), 1]), ...
                      RESHAPE(f([size(obj.a, 4), size(obj_low.a, 4)]), [1, size(obj.a, 4), size(obj_low.a, 4), 1]), ...
                      RESHAPE(f([size(obj.a, 1), size(obj_low.a, 1)]), [1, size(obj.a, 1), size(obj_low.a, 1), 1]), };
  
  
            obj.TB = {RESHAPE(f([size(obj.b, 2), size(obj_low.b, 2)]), [1, size(obj.b, 2), size(obj_low.b, 2), 1]), ...
                      RESHAPE(f([size(obj.b, 3), size(obj_low.b, 3)]), [1, size(obj.b, 3), size(obj_low.b, 3), 1]), ...
                      RESHAPE(f([size(obj.b, 4), size(obj_low.b, 4)]), [1, size(obj.b, 4), size(obj_low.b, 4), 1]), ...
                      RESHAPE(f([size(obj.b, 1), size(obj_low.b, 1)]), [1, size(obj.b, 1), size(obj_low.b, 1), 1]), };

        end
        

        function leftAB = CTMContraction(~, a, a_low_conj, TA, CA, indices_order, mode)
            
            tempa = permute(a, indices_order);
            tempa_low_conj = permute(a_low_conj, indices_order);

            leftAB = tensorprod(TA{indices_order(4)}, ...
                                CA{indices_order(1)}, 4, 1, NumDimensionsA=4);

            leftAB = tensorprod(leftAB, TA{indices_order(1)}, 4, 1, NumDimensionsA=4);

            leftAB = tensorprod(leftAB, tempa, [2, 4], [1, 2], NumDimensionsA=6);
            leftAB = permute(leftAB, [1, 4, 2, 3, 5, 6, 7]);

            if mode == 1
                % without fusing the physical indices
                leftAB = tensorprod(leftAB, tempa_low_conj, [3, 4], [1, 2], NumDimensionsA=7);
                leftAB = permute(leftAB, [1, 4, 7, 2, 3, 6, 5, 8]);
                return
            end

            leftAB = tensorprod(leftAB, tempa_low_conj, [3, 4, 7], [1, 2, 5], NumDimensionsA=7);

            if mode == 0
                leftAB = permute(leftAB, [1, 4, 6, 2, 3, 5]);
                leftAB = RESHAPE(leftAB, ...
                    [size(TA{indices_order(4)}, 1) * size(tempa, 4) * size(tempa_low_conj, 4), ...
                    size(TA{indices_order(1)}, 4) * size(tempa, 3) * size(tempa_low_conj, 3)]);
            elseif mode == 2
                leftAB = RESHAPE(leftAB, [sqrt(size(TA{indices_order(4)}, 1)), sqrt(size(TA{indices_order(4)}, 1)), ...
                                        sqrt(size(TA{indices_order(1)}, 4)), sqrt(size(TA{indices_order(1)}, 4)), ...
                                        size(tempa, 3), size(tempa, 4), ...
                                        size(tempa, 3), size(tempa, 4)]);
                leftAB = permute(leftAB, [1, 6, 3, 5, 2, 8, 4, 7]);
                leftAB = RESHAPE(leftAB, ...
                [size(leftAB, 1) * size(leftAB, 2) * size(leftAB, 3) * size(leftAB, 4), ...
                 size(leftAB, 5) * size(leftAB, 6) * size(leftAB, 7) * size(leftAB, 8)]);
            end
        end

        function temp = CTMCornerUpdate(~, C, T, P, direction)
            if direction == 0
                temp = tensorprod(T, C, 4, 1, NumDimensionsA=4);
                temp = tensorprod(temp, P, [2, 3, 4], [2, 3, 1], NumDimensionsA=4);
                return 
            elseif direction == 1
                temp = tensorprod(C, T, 2, 1, NumDimensionsA=2);
                temp = tensorprod(P, temp, [1, 2, 3], [1, 2, 3], NumDimensionsA=4);
                return
            end
        end

        function temp = CTMEdgeUpdate(~, Pright, T, a, a_low_conj, Pleft)
            temp = tensorprod(Pright, T, 1, 1, NumDimensionsA=4);
            temp = permute(temp, [3, 1, 2, 4, 5, 6]);
            temp = tensorprod(temp, a, [2, 4], [1, 2], NumDimensionsA=6);
            temp = tensorprod(temp, a_low_conj, [2, 3, 7], [1, 2, 5], NumDimensionsA=7);
            temp = tensorprod(temp, Pleft, [2, 3, 5], [1, 2, 3], NumDimensionsA=6);
        end

    

        function corners = CTMRenormCornersQR(obj, obj_low, bond_dir)

            LIO = [mod([0, 1, 2, 3] + bond_dir, 4) + 1, 5];
            RIO = [mod([1, 2, 3, 0] + bond_dir, 4) + 1, 5];

            LIO_ = [mod([0, 1, 2, 3] + bond_dir + 2, 4) + 1, 5];
            RIO_ = [mod([1, 2, 3, 0] + bond_dir + 2, 4) + 1, 5];

            leftAB_up = obj.CTMContraction(obj.a, conj(obj_low.a), obj.TA, obj.CA, LIO, 0);
            rightAB_up = obj.CTMContraction(obj.b, conj(obj_low.b), obj.TB, obj.CB, RIO, 0);

            leftAB_below = obj.CTMContraction(obj.a, conj(obj_low.a), obj.TA, obj.CA, LIO_, 0);
            rightAB_below = obj.CTMContraction(obj.b, conj(obj_low.b), obj.TB, obj.CB, RIO_, 0);

            leftBA_up = obj.CTMContraction(obj.b, conj(obj_low.b), obj.TB, obj.CB, LIO, 0);
            rightBA_up = obj.CTMContraction(obj.a, conj(obj_low.a), obj.TA, obj.CA, RIO, 0);

            leftBA_below = obj.CTMContraction(obj.b, conj(obj_low.b), obj.TB, obj.CB, LIO_, 0);
            rightBA_below = obj.CTMContraction(obj.a, conj(obj_low.a), obj.TA, obj.CA, RIO_, 0);

            corners = {leftAB_up, rightAB_up, leftBA_up, rightBA_up, leftAB_below, rightAB_below, leftBA_below, rightBA_below};

        end

        function result = CutOffSum(~, a, tol)
            temp = zeros(1, length(a));
            temp(1) = a(1);
            for ii = 2:length(a)
                temp(ii) = temp(ii - 1) + a(ii);
            end
            temp = temp / sum(a);
            result = max(1, sum((1 - temp) > tol));
        end
        
        function EnlargeCTM(obj)
            
            new_CA = {0, 0, 0, 0,};
            new_CB = {0, 0, 0, 0,};
            new_TA = {0, 0, 0, 0,};
            new_TB = {0, 0, 0, 0,};

            for bond_dir = 0:3
                
                LIO = [mod([0, 1, 2, 3] + bond_dir, 4) + 1, 5];
                
                new_CB{bond_dir + 1} = obj.CTMContraction(obj.a, conj(obj.a), obj.TA, obj.CA, LIO, 0);
                new_CA{bond_dir + 1} = obj.CTMContraction(obj.b, conj(obj.b), obj.TB, obj.CB, LIO, 0);
                
                new_TB{bond_dir + 1} = tensorprod(obj.TA{bond_dir + 1}, obj.a, 2, LIO(2));
                new_TB{bond_dir + 1} = tensorprod(new_TB{bond_dir + 1}, conj(obj.a), [2, 7], [LIO(2), 5]);
                new_TB{bond_dir + 1} = RESHAPE(permute(new_TB{bond_dir + 1}, [1, 3, 6, 5, 8, 2, 4, 7]), ...
                [size(obj.TA{bond_dir + 1}, 1) * size(obj.a, LIO(1)) * size(obj.a, LIO(1)), ...
                    size(obj.a, LIO(4)), size(obj.a, LIO(4)), size(obj.TA{bond_dir + 1}, 4) * size(obj.a, LIO(3)) * size(obj.a, LIO(3))]);
                
                new_TA{bond_dir + 1} = tensorprod(obj.TB{bond_dir + 1}, obj.b, 2, LIO(2));
                new_TA{bond_dir + 1} = tensorprod(new_TA{bond_dir + 1}, conj(obj.b), [2, 7], [LIO(2), 5]);
                new_TA{bond_dir + 1} = RESHAPE(permute(new_TA{bond_dir + 1}, [1, 3, 6, 5, 8, 2, 4, 7]), ...
                [size(obj.TB{bond_dir + 1}, 1) * size(obj.b, LIO(1)) * size(obj.b, LIO(1)), ...
                    size(obj.b, LIO(4)), size(obj.b, LIO(4)), size(obj.TB{bond_dir + 1}, 4) * size(obj.b, LIO(3)) * size(obj.b, LIO(3))]);

            end
            obj.TA = new_TA;
            obj.TB = new_TB;
            obj.CA = new_CA;
            obj.CB = new_CB;
        end

        function CTMRenormSingleLayerQR(obj, obj_low, corners, bond_dir, ctm_svd_tol, normalize, update_now)

            LIO = [mod([0, 1, 2, 3] + bond_dir, 4) + 1, 5];
            RIO = [mod([1, 2, 3, 0] + bond_dir, 4) + 1, 5];

            leftAB_up = corners{1};
            rightAB_up = corners{2};
            leftBA_up = corners{3};
            rightBA_up = corners{4};
            leftAB_below = corners{5};
            rightAB_below = corners{6};
            leftBA_below = corners{7};
            rightBA_below = corners{8};

            AB_up_left = rightAB_below * leftAB_up;
            AB_up_left = RESHAPE(AB_up_left, [size(rightAB_below, 1), ...
                                              sqrt(size(obj.TA{LIO(1)}, 4)), sqrt(size(obj.TA{LIO(1)}, 4)), ...
                                              size(obj.a, LIO(3)), size(obj.a, LIO(3))]); 
            AB_up_left = permute(AB_up_left, [1, 3, 5, 2, 4]);
            AB_up_left = RESHAPE(AB_up_left, [size(rightAB_below, 1) * sqrt(size(obj.TA{LIO(1)}, 4)) * size(obj.a, LIO(3)), ...
                                              sqrt(size(obj.TA{LIO(1)}, 4)) * size(obj.a, LIO(3))]);
            RAB_up_left = qr(AB_up_left, "econ");

            AB_up_right = rightAB_up * leftAB_below;
            AB_up_right = RESHAPE(AB_up_right, [sqrt(size(obj.TB{LIO(1)}, 1)), sqrt(size(obj.TB{LIO(1)}, 1)), ...
                                                size(obj.b, LIO(1)), size(obj.b, LIO(1)), ...
                                                size(leftAB_below, 2)]); 
            AB_up_right = permute(AB_up_right, [1, 3, 5, 2, 4]);
            AB_up_right = RESHAPE(AB_up_right, [sqrt(size(obj.TB{LIO(1)}, 1)) * size(obj.b, LIO(1)), ...
                                                size(leftAB_below, 2) * sqrt(size(obj.TB{LIO(1)}, 1)) * size(obj.b, LIO(1))]);
            RAB_up_right = transpose(qr(transpose(AB_up_right), "econ"));

            BA_up_left = rightBA_below * leftBA_up;
            BA_up_left = RESHAPE(BA_up_left, [size(rightBA_below, 1), ...
                                              sqrt(size(obj.TB{LIO(1)}, 4)), sqrt(size(obj.TB{LIO(1)}, 4)), ...
                                              size(obj.b, LIO(3)), size(obj.b, LIO(3))]); 
            BA_up_left = permute(BA_up_left, [1, 3, 5, 2, 4]);
            BA_up_left = RESHAPE(BA_up_left, [size(rightBA_below, 1) * sqrt(size(obj.TB{LIO(1)}, 4)) * size(obj.b, LIO(3)), ...
                                              sqrt(size(obj.TB{LIO(1)}, 4)) * size(obj.b, LIO(3))]);
            RBA_up_left = qr(BA_up_left, "econ");

            BA_up_right = rightBA_up * leftBA_below;
            BA_up_right = RESHAPE(BA_up_right, [sqrt(size(obj.TA{LIO(1)}, 1)), sqrt(size(obj.TA{LIO(1)}, 1)), ...
                                                size(obj.a, LIO(1)), size(obj.a, LIO(1)), ...
                                                size(leftBA_below, 2)]); 
            BA_up_right = permute(BA_up_right, [1, 3, 5, 2, 4]);
            BA_up_right = RESHAPE(BA_up_right, [sqrt(size(obj.TA{LIO(1)}, 1)) * size(obj.a, LIO(1)), ...
                                                size(leftBA_below, 2) * sqrt(size(obj.TA{LIO(1)}, 1)) * size(obj.a, LIO(1))]);
            RBA_up_right = transpose(qr(transpose(BA_up_right), "econ"));

            RAB_0 = RAB_up_left * RAB_up_right;
            RBA_0 = RBA_up_left * RBA_up_right;


            [UAB, SAB, VAB] = svd(RAB_0, "econ");
            [UBA, SBA, VBA] = svd(RBA_0, "econ");
            

            SAB_cut = diag(SAB);
            new_chi_AB = min(sum((SAB_cut / SAB_cut(1))> ctm_svd_tol), ...
                                                floor(sqrt(obj.chimax)));
            % fprintf("Singular values of AB: \n")
            % fprintf("%.9e ", SAB_cut);
            % fprintf("\n")
            SBA_cut = diag(SBA);
            new_chi_BA = min(sum((SBA_cut / SBA_cut(1))> ctm_svd_tol), ...
                                                floor(sqrt(obj.chimax)));
            % fprintf("Singular values of BA: \n")
            % fprintf("%.9e ", SBA_cut);
            % fprintf("\n")
                

            while true
                % % fprintf("Singular values of AB: \n")
                
                SAB_cut = diag(SAB);
                % fprintf("SVD\n")
                % fprintf("bond AB SVD error: %.9e\n", 1 - sum(SAB_cut(1:new_chi_AB)) / sum(SAB_cut));
                SAB_cut = 1.0 ./ sqrt(SAB_cut(1:new_chi_AB));
                % fprintf('Building Projectors for bond AB\n');
            

                VABh = VAB';
                UABh = UAB';

                diag_sqrt_inv_SAB_cut = diag(SAB_cut);

                PAB_left = RAB_up_right * VAB(:, 1:new_chi_AB) * diag_sqrt_inv_SAB_cut;
                PAB_right = transpose(diag_sqrt_inv_SAB_cut * UABh(1:new_chi_AB, :) * RAB_up_left);
                % PAB_right = transpose(pinv(PAB_left));
                eig_AB = eig(PAB_left * transpose(PAB_right));
                % fprintf("CTM error: %.9e\n", sum(diag(RAB_0 - RAB_up_left * PAB_left * transpose(PAB_right) * RAB_up_right)) / sum(diag(RAB_0)))
                err_1 = [];
                err_0 = [];
                eig_AB = transpose(eig_AB);
                for eig_ = eig_AB
                    abs_eig_ = abs(eig_);
                    if abs_eig_ > 0.5
                        err_1 = [err_1, eig_];
                    else
                        err_0 = [err_0, eig_];
                    end
                end
                errAB = max([abs(err_1 - 1), abs(err_0)]);
                % fprintf("Projector AB error: %.9e\n", errAB);

                if (errAB < 1e-6)
                    break
                elseif new_chi_AB == 1
                    break
                else
                    new_chi_AB = new_chi_AB - 1;
                end
            end
            
            
            while true

                SBA_cut = diag(SBA);

                % fprintf("bond BA SVD error: %.9e\n", 1 - sum(SBA_cut(1:new_chi_BA)) / sum(SBA_cut));
                SBA_cut = 1.0 ./ sqrt(SBA_cut(1:new_chi_BA));
                % fprintf('Building Projectors for bond BA\n');

                VBAh = VBA';
                UBAh = UBA';
                
                diag_sqrt_inv_SBA_cut = diag(SBA_cut);

                PBA_left = RBA_up_right * VBA(:, 1:new_chi_BA) * diag_sqrt_inv_SBA_cut;
                PBA_right = transpose(diag_sqrt_inv_SBA_cut * UBAh(1:new_chi_BA, :) * RBA_up_left);
                % PBA_right = transpose(pinv(PBA_left));
                test_mat = PBA_left * transpose(PBA_right);
                eig_BA = eig(test_mat);
                % fprintf("CTM error: %.9e\n", sum(diag(RBA_0 - RBA_up_left * PBA_left * transpose(PBA_right) * RBA_up_right)) / sum(diag(RBA_0)))
                err_1 = [];
                err_0 = [];
                eig_BA = transpose(eig_BA);
                for eig_ = eig_BA
                    abs_eig = abs(eig_);
                    if abs_eig > 0.5
                        err_1 = [err_1, eig_];
                    else
                        err_0 = [err_0, eig_];
                    end
                end
                % PBA_right = PBA_right_old;
                errBA = max([abs(err_1 - 1), abs(err_0)]);
                % fprintf("Projector BA error: %.9e\n", errBA);

                if (errBA < 1e-6)
                    break
                elseif new_chi_BA == 1
                    break
                else
                    new_chi_BA = new_chi_BA - 1;
                end

            end

            PAB_left = kron(PAB_left, conj(PAB_left));
            PAB_left = RESHAPE(PAB_left, [sqrt(size(obj.TA{LIO(1)}, 4)), size(obj.a, LIO(3)), sqrt(size(obj.TA{LIO(1)}, 4)), size(obj.a, LIO(3)), size(PAB_left, 2)]);
            PAB_left = permute(PAB_left, [1, 3, 2, 4, 5]);
            PAB_left = RESHAPE(PAB_left, [size(obj.TA{LIO(1)}, 4) * size(obj.a, LIO(3)) * size(obj.a, LIO(3)), size(PAB_left, 5)]);

            PBA_left = kron(PBA_left, conj(PBA_left));
            PBA_left = RESHAPE(PBA_left, [sqrt(size(obj.TB{LIO(1)}, 4)), size(obj.b, LIO(3)), sqrt(size(obj.TB{LIO(1)}, 4)), size(obj.b, LIO(3)), size(PBA_left, 2)]);
            PBA_left = permute(PBA_left, [1, 3, 2, 4, 5]);
            PBA_left = RESHAPE(PBA_left, [size(obj.TB{LIO(1)}, 4) * size(obj.b, LIO(3)) * size(obj.b, LIO(3)), size(PBA_left, 5)]);

            PAB_right = kron(PAB_right, conj(PAB_right));
            PAB_right = RESHAPE(PAB_right, [sqrt(size(obj.TB{LIO(1)}, 1)), size(obj.b, LIO(1)), sqrt(size(obj.TB{LIO(1)}, 1)), size(obj.b, LIO(1)), size(PAB_right, 2)]);
            PAB_right = permute(PAB_right, [1, 3, 2, 4, 5]);
            PAB_right = RESHAPE(PAB_right, [size(obj.TB{LIO(1)}, 1) * size(obj.b, LIO(1)) * size(obj.b, LIO(1)), size(PAB_right, 5)]);

            PBA_right = kron(PBA_right, conj(PBA_right));
            PBA_right = RESHAPE(PBA_right, [sqrt(size(obj.TA{LIO(1)}, 1)), size(obj.a, LIO(1)), sqrt(size(obj.TA{LIO(1)}, 1)), size(obj.a, LIO(1)), size(PBA_right, 2)]);
            PBA_right = permute(PBA_right, [1, 3, 2, 4, 5]);
            PBA_right = RESHAPE(PBA_right, [size(obj.TA{LIO(1)}, 1) * size(obj.a, LIO(1)) * size(obj.a, LIO(1)), size(PBA_right, 5)]);
            
            PAB_left = RESHAPE(PAB_left, ...
                [size(obj.TA{LIO(1)}, 4), size(obj.a, LIO(3)), size(obj_low.a, LIO(3)),  new_chi_AB ^ 2]);

            PAB_right = RESHAPE(PAB_right, ...
                [size(obj.TB{LIO(1)}, 1), size(obj.b, LIO(1)), size(obj_low.b, LIO(1)),  new_chi_AB ^ 2]);

            % fprintf("Projectors for bond AB : Built, new_chi_AB: %d\n", new_chi_AB)

            PBA_left = RESHAPE(PBA_left, ...
                [size(obj.TB{LIO(1)}, 4), size(obj.b, LIO(3)), size(obj_low.b, LIO(3)), new_chi_BA ^ 2]);
            PBA_right = RESHAPE(PBA_right, ...
                [size(obj.TA{LIO(1)}, 1), size(obj.a, LIO(1)), size(obj_low.a, LIO(1)), new_chi_BA ^ 2]);

            % fprintf("Projectors for bond BA: Built, new_chi_BA: %d\n", new_chi_BA)

            CB_temp_0 = obj.CTMCornerUpdate(obj.CA{LIO(1)}, obj.TA{LIO(4)}, PBA_left, 0);
            CA_temp_0 = obj.CTMCornerUpdate(obj.CB{LIO(1)}, obj.TB{LIO(4)}, PAB_left, 0);

            CB_temp_1 = obj.CTMCornerUpdate(obj.CA{LIO(2)}, obj.TA{LIO(2)}, PAB_right, 1);
            CA_temp_1 = obj.CTMCornerUpdate(obj.CB{LIO(2)}, obj.TB{LIO(2)}, PBA_right, 1);
    
            TB_temp = obj.CTMEdgeUpdate(PBA_right, obj.TA{LIO(1)}, permute(obj.a, LIO), conj(permute(obj_low.a, LIO)), PAB_left);
            TA_temp = obj.CTMEdgeUpdate(PAB_right, obj.TB{LIO(1)}, permute(obj.b, LIO), conj(permute(obj_low.b, LIO)), PBA_left);

            % fprintf("CTM updated\n")

            if update_now ~= 1
                return
            end

            if normalize == 1
                obj.CA{LIO(1)} = CA_temp_0 / norm(CA_temp_0(:));
                obj.CB{LIO(1)} = CB_temp_0 / norm(CB_temp_0(:));

                obj.CA{LIO(2)} = CA_temp_1 / norm(CA_temp_1(:));
                obj.CB{LIO(2)} = CB_temp_1 / norm(CB_temp_1(:));

                obj.TA{LIO(1)} = TA_temp / norm(TA_temp(:));
                obj.TB{LIO(1)} = TB_temp / norm(TB_temp(:));
                % fprintf("Normalized!\n")
            else
                obj.CA{LIO(1)} = CA_temp_0;
                obj.CB{LIO(1)} = CB_temp_0;

                obj.CA{LIO(2)} = CA_temp_1;
                obj.CB{LIO(2)} = CB_temp_1;

                obj.TA{LIO(1)} = TA_temp;
                obj.TB{LIO(1)} = TB_temp;
            end
        end

        function CTMRenorm_QR_test(obj, obj_low, corners, bond_dir, ctm_svd_tol, normalize, update_now, InitVUMPS)

            LIO = [mod([0, 1, 2, 3] + bond_dir, 4) + 1, 5];

            new_chi_AB = size(obj.a, LIO(3)) * size(obj.a, LIO(3));
            new_chi_BA = size(obj.b, LIO(3)) * size(obj.b, LIO(3));


            PAB_left = eye(size(obj.a, LIO(3)) * size(obj.a, LIO(3)), size(obj.a, LIO(3)) * size(obj.a, LIO(3)));
            PAB_right = eye(size(obj.a, LIO(3)) * size(obj.a, LIO(3)), size(obj.a, LIO(3)) * size(obj.a, LIO(3)));
            PBA_left = eye(size(obj.b, LIO(3)) * size(obj.b, LIO(3)), size(obj.b, LIO(3)) * size(obj.b, LIO(3)));
            PBA_right = eye(size(obj.b, LIO(3)) * size(obj.b, LIO(3)), size(obj.b, LIO(3)) * size(obj.b, LIO(3)));
            
            PAB_left = RESHAPE(PAB_left, ...
                [size(obj.TA{LIO(1)}, 4), size(obj.a, LIO(3)), size(obj_low.a, LIO(3)),  new_chi_AB]);

            PAB_right = RESHAPE(PAB_right, ...
                [size(obj.TB{LIO(1)}, 1), size(obj.b, LIO(1)), size(obj_low.b, LIO(1)),  new_chi_AB]);

            % fprintf("Projectors for bond AB : Built, new_chi_AB: %d\n", new_chi_AB)

            PBA_left = RESHAPE(PBA_left, ...
                [size(obj.TB{LIO(1)}, 4), size(obj.b, LIO(3)), size(obj_low.b, LIO(3)), new_chi_BA]);
            PBA_right = RESHAPE(PBA_right, ...
                [size(obj.TA{LIO(1)}, 1), size(obj.a, LIO(1)), size(obj_low.a, LIO(1)), new_chi_BA]);

            % fprintf("Projectors for bond BA: Built, new_chi_BA: %d\n", new_chi_BA)

            CB_temp_0 = obj.CTMCornerUpdate(obj.CA{LIO(1)}, obj.TA{LIO(4)}, PBA_left, 0);
            CA_temp_0 = obj.CTMCornerUpdate(obj.CB{LIO(1)}, obj.TB{LIO(4)}, PAB_left, 0);

            CB_temp_1 = obj.CTMCornerUpdate(obj.CA{LIO(2)}, obj.TA{LIO(2)}, PAB_right, 1);
            CA_temp_1 = obj.CTMCornerUpdate(obj.CB{LIO(2)}, obj.TB{LIO(2)}, PBA_right, 1);
    
            TB_temp = obj.CTMEdgeUpdate(PBA_right, obj.TA{LIO(1)}, permute(obj.a, LIO), conj(permute(obj_low.a, LIO)), PAB_left);
            TA_temp = obj.CTMEdgeUpdate(PAB_right, obj.TB{LIO(1)}, permute(obj.b, LIO), conj(permute(obj_low.b, LIO)), PBA_left);

            if update_now ~= 1
                return
            end

            if normalize == 1
                obj.CA{LIO(1)} = CA_temp_0 / norm(CA_temp_0(:));
                obj.CB{LIO(1)} = CB_temp_0 / norm(CB_temp_0(:));

                obj.CA{LIO(2)} = CA_temp_1 / norm(CA_temp_1(:));
                obj.CB{LIO(2)} = CB_temp_1 / norm(CB_temp_1(:));

                obj.TA{LIO(1)} = TA_temp / norm(TA_temp(:));
                obj.TB{LIO(1)} = TB_temp / norm(TB_temp(:));
            else
                obj.CA{LIO(1)} = CA_temp_0;
                obj.CB{LIO(1)} = CB_temp_0;

                obj.CA{LIO(2)} = CA_temp_1;
                obj.CB{LIO(2)} = CB_temp_1;

                obj.TA{LIO(1)} = TA_temp;
                obj.TB{LIO(1)} = TB_temp;
            end
        end

        function CTMRenorm_QR_(obj, obj_low, corners, bond_dir, ctm_svd_tol, normalize, update_now, InitVUMPS)

            LIO = [mod([0, 1, 2, 3] + bond_dir, 4) + 1, 5];

            leftAB_up = corners{1};
            rightAB_up = corners{2};
            leftBA_up = corners{3};
            rightBA_up = corners{4};
            leftAB_below = corners{5};
            rightAB_below = corners{6};
            leftBA_below = corners{7};
            rightBA_below = corners{8};

            RAB_up_left = qr(rightAB_below * leftAB_up, "econ");
            RAB_up_right = transpose(qr(transpose(rightAB_up * leftAB_below), "econ"));
            RBA_up_left = qr(rightBA_below * leftBA_up, "econ");
            RBA_up_right = transpose(qr(transpose(rightBA_up * leftBA_below), "econ"));


            RAB_0 = RAB_up_left * RAB_up_right;
            RBA_0 = RBA_up_left * RBA_up_right;

            % [UAB, SAB, VAB] = svds(RAB_0, obj.chimax);
            % [UBA, SBA, VBA] = svds(RBA_0, obj.chimax);

            [UAB, SAB, VAB] = svd(RAB_0, "econ");
            [UBA, SBA, VBA] = svd(RBA_0, "econ");
            

            SAB_cut = diag(SAB);

            if InitVUMPS == 1
                new_chi_AB = 1;
            else
                new_chi_AB = min(sum((SAB_cut / SAB_cut(1)> ctm_svd_tol)), ...
                                    obj.chimax);
            end
            % % disp(ctm_svd_tol)
            % fprintf("Initial new_chi_AB: %d\n", new_chi_AB)
            % % fprintf("Singular values of AB: \n")
            % % fprintf("%.9e ", SAB_cut);
            % % fprintf("\n")

            SBA_cut = diag(SBA);
            if InitVUMPS == 1
                new_chi_BA = 1;
            else
                new_chi_BA = min(sum((SBA_cut / SBA_cut(1)> ctm_svd_tol)), ...
                                    obj.chimax);
            end

            % % fprintf("Singular values of BA: \n")
            % % fprintf("%.9e ", SBA_cut);
            % % fprintf("\n")
            % fprintf("Initial new_chi_BA: %d\n", new_chi_BA)
                

            while true
                % % fprintf("Singular values of AB: \n")
                
                SAB_cut = diag(SAB);
                % fprintf("SVD\n")
                % fprintf("bond AB SVD error: %.9e\n", 1 - sum(SAB_cut(1:new_chi_AB)) / sum(SAB_cut));
                SAB_cut = 1.0 ./ sqrt(SAB_cut(1:new_chi_AB));
                % fprintf('Building Projectors for bond AB\n');
            

                VABh = VAB';
                UABh = UAB';

                diag_sqrt_inv_SAB_cut = diag(SAB_cut);

                PAB_left = RAB_up_right * VAB(:, 1:new_chi_AB) * diag_sqrt_inv_SAB_cut;

                PAB_right_test = transpose(diag_sqrt_inv_SAB_cut * UABh(1:new_chi_AB, :) * RAB_up_left);
                PAB_right = transpose(pinv(PAB_left, 1e-10));

                % fprintf("Difference between projectors AB: %.9e\n", norm(PAB_right_test - PAB_right, "fro"))

                eig_AB = eig(PAB_left * transpose(PAB_right));
                % fprintf("CTM error: %.9e\n", trace(RAB_0 - RAB_up_left * PAB_left * transpose(PAB_right) * RAB_up_right) / trace(RAB_0))
                err_1 = [];
                err_0 = [];
                eig_AB = transpose(eig_AB);
                for eig_ = eig_AB
                    abs_eig_ = abs(eig_);
                    if abs_eig_ > 0.5
                        err_1 = [err_1, eig_];
                    else
                        err_0 = [err_0, eig_];
                    end
                end
                errAB = max([abs(err_1 - 1), abs(err_0)]);
                % fprintf("Projector AB error: %.9e\n", errAB);

                % break;

                if (errAB < 1e-6)
                    break
                elseif new_chi_AB == 1
                    break
                else
                    new_chi_AB = new_chi_AB - 1;
                end
                
            end
            
            
            while true

                SBA_cut = diag(SBA);

                % fprintf("bond BA SVD error: %.9e\n", 1 - sum(SBA_cut(1:new_chi_BA)) / sum(SBA_cut));
                SBA_cut = 1.0 ./ sqrt(SBA_cut(1:new_chi_BA));
                % fprintf('Building Projectors for bond BA\n');

                VBAh = VBA';
                UBAh = UBA';
                
                diag_sqrt_inv_SBA_cut = diag(SBA_cut);

                PBA_left = RBA_up_right * VBA(:, 1:new_chi_BA) * diag_sqrt_inv_SBA_cut;

                PBA_right_test = transpose(diag_sqrt_inv_SBA_cut * UBAh(1:new_chi_BA, :) * RBA_up_left);
                PBA_right = transpose(pinv(PBA_left, 1e-10));

                % fprintf("Difference between projectors BA: %.9e\n", norm(PBA_right_test - PBA_right, "fro"))
                test_mat = PBA_left * transpose(PBA_right);
                eig_BA = eig(test_mat);
                % fprintf("CTM error: %.9e\n", trace(RBA_0 - RBA_up_left * PBA_left * transpose(PBA_right) * RBA_up_right) / trace(RBA_0))
                err_1 = [];
                err_0 = [];
                eig_BA = transpose(eig_BA);
                for eig_ = eig_BA
                    abs_eig = abs(eig_);
                    if abs_eig > 0.5
                        err_1 = [err_1, eig_];
                    else
                        err_0 = [err_0, eig_];
                    end
                end
                % PBA_right = PBA_right_old;
                errBA = max([abs(err_1 - 1), abs(err_0)]);
                % fprintf("Projector BA error: %.9e\n", errBA);

                % break;

                if (errBA < 1e-6)
                    break
                elseif new_chi_BA == 1
                    break
                else
                    new_chi_BA = new_chi_BA - 1;
                end

            end
            
            PAB_left = RESHAPE(PAB_left, ...
                [size(obj.TA{LIO(1)}, 4), size(obj.a, LIO(3)), size(obj_low.a, LIO(3)),  new_chi_AB]);

            PAB_right = RESHAPE(PAB_right, ...
                [size(obj.TB{LIO(1)}, 1), size(obj.b, LIO(1)), size(obj_low.b, LIO(1)),  new_chi_AB]);

            % fprintf("Projectors for bond AB : Built, new_chi_AB: %d\n", new_chi_AB)

            PBA_left = RESHAPE(PBA_left, ...
                [size(obj.TB{LIO(1)}, 4), size(obj.b, LIO(3)), size(obj_low.b, LIO(3)), new_chi_BA]);
            PBA_right = RESHAPE(PBA_right, ...
                [size(obj.TA{LIO(1)}, 1), size(obj.a, LIO(1)), size(obj_low.a, LIO(1)), new_chi_BA]);

            % fprintf("Projectors for bond BA: Built, new_chi_BA: %d\n", new_chi_BA)

            CB_temp_0 = obj.CTMCornerUpdate(obj.CA{LIO(1)}, obj.TA{LIO(4)}, PBA_left, 0);
            CA_temp_0 = obj.CTMCornerUpdate(obj.CB{LIO(1)}, obj.TB{LIO(4)}, PAB_left, 0);

            CB_temp_1 = obj.CTMCornerUpdate(obj.CA{LIO(2)}, obj.TA{LIO(2)}, PAB_right, 1);
            CA_temp_1 = obj.CTMCornerUpdate(obj.CB{LIO(2)}, obj.TB{LIO(2)}, PBA_right, 1);
    
            TB_temp = obj.CTMEdgeUpdate(PBA_right, obj.TA{LIO(1)}, permute(obj.a, LIO), conj(permute(obj_low.a, LIO)), PAB_left);
            TA_temp = obj.CTMEdgeUpdate(PAB_right, obj.TB{LIO(1)}, permute(obj.b, LIO), conj(permute(obj_low.b, LIO)), PBA_left);

            if update_now ~= 1
                return
            end

            if normalize == 1
                obj.CA{LIO(1)} = CA_temp_0 / norm(CA_temp_0(:));
                obj.CB{LIO(1)} = CB_temp_0 / norm(CB_temp_0(:));

                obj.CA{LIO(2)} = CA_temp_1 / norm(CA_temp_1(:));
                obj.CB{LIO(2)} = CB_temp_1 / norm(CB_temp_1(:));

                obj.TA{LIO(1)} = TA_temp / norm(TA_temp(:));
                obj.TB{LIO(1)} = TB_temp / norm(TB_temp(:));
            else
                obj.CA{LIO(1)} = CA_temp_0;
                obj.CB{LIO(1)} = CB_temp_0;

                obj.CA{LIO(2)} = CA_temp_1;
                obj.CB{LIO(2)} = CB_temp_1;

                obj.TA{LIO(1)} = TA_temp;
                obj.TB{LIO(1)} = TB_temp;
            end
        end

        function CTMRenorm_QR_test_inv(obj, obj_low, corners, bond_dir, ctm_svd_tol, normalize, update_now, InitVUMPS)

            LIO = [mod([0, 1, 2, 3] + bond_dir, 4) + 1, 5];

            leftAB_up = corners{1};
            rightAB_up = corners{2};
            leftBA_up = corners{3};
            rightBA_up = corners{4};
            leftAB_below = corners{5};
            rightAB_below = corners{6};
            leftBA_below = corners{7};
            rightBA_below = corners{8};

            RAB_up_left = qr(rightAB_below * leftAB_up, "econ");
            RAB_up_right = transpose(qr(transpose(rightAB_up * leftAB_below), "econ"));
            RBA_up_left = qr(rightBA_below * leftBA_up, "econ");
            RBA_up_right = transpose(qr(transpose(rightBA_up * leftBA_below), "econ"));


            RAB_0 = RAB_up_left * RAB_up_right;
            RBA_0 = RBA_up_left * RBA_up_right;

            % [UAB, SAB, VAB] = svds(RAB_0, obj.chimax);
            % [UBA, SBA, VBA] = svds(RBA_0, obj.chimax);

            % [UAB, SAB, VAB] = svd(RAB_0, "econ");
            % [UBA, SBA, VBA] = svd(RBA_0, "econ");
            
            [VAB, SAB_, UAB] = svd(pinv(RAB_0), "econ");
            [VBA, SBA_, UBA] = svd(pinv(RBA_0), "econ");
            

            SAB_cut = diag(SAB_);
            SBA_cut = diag(SBA_);

            new_chi_AB = min(obj.chimax, length(SAB_cut));
            new_chi_BA = min(obj.chimax, length(SAB_cut));

            % if InitVUMPS == 1
            %     new_chi_AB = 1;
            % else
            %     new_chi_AB = min(sum((SAB_cut / SAB_cut(1)> ctm_svd_tol)), ...
            %                         obj.chimax);
            % end
            % % fprintf("Singular values of AB: \n")
            % % fprintf("%.9e ", SAB_cut);
            % % fprintf("\n")

            % SBA_cut = diag(SBA);
            % if InitVUMPS == 1
            %     new_chi_BA = 1;
            % else
            %     new_chi_BA = min(sum((SBA_cut / SBA_cut(1)> ctm_svd_tol)), ...
            %                         obj.chimax);
            % end
            % % fprintf("Singular values of BA: \n")
            % % fprintf("%.9e ", SBA_cut);
            % % fprintf("\n")
                

            while true
                % % fprintf("Singular values of AB: \n")
                
                % SAB_cut = diag(SAB);
                % % fprintf("SVD\n")
                % % fprintf("bond AB SVD error: %.9e\n", 1 - sum(SAB_cut(1:new_chi_AB)) / sum(SAB_cut));
                % SAB_cut = 1.0 ./ sqrt(SAB_cut(1:new_chi_AB));
                
                SAB_cut = sqrt(SAB_cut(length(SAB_cut) - new_chi_AB + 1:length(SAB_cut)));
                % fprintf('Building Projectors for bond AB\n');
            

                VABh = VAB';
                UABh = UAB';

                diag_sqrt_inv_SAB_cut = diag(SAB_cut);

                % PAB_left = RAB_up_right * VAB(:, 1:new_chi_AB) * diag_sqrt_inv_SAB_cut;
                % PAB_right = transpose(diag_sqrt_inv_SAB_cut * UABh(1:new_chi_AB, :) * RAB_up_left);
                PAB_left = RAB_up_right * VAB(:, length(SAB_cut) - new_chi_AB + 1:length(SAB_cut)) * diag_sqrt_inv_SAB_cut;
                PAB_right = transpose(diag_sqrt_inv_SAB_cut * UABh(length(SAB_cut) - new_chi_AB + 1:length(SAB_cut), :) * RAB_up_left);
                % PAB_right = transpose(pinv(PAB_left));

                eig_AB = eig(PAB_left * transpose(PAB_right));
                % fprintf("CTM error: %.9e\n", trace(RAB_0 - RAB_up_left * PAB_left * transpose(PAB_right) * RAB_up_right) / trace(RAB_0))
                err_1 = [];
                err_0 = [];
                eig_AB = transpose(eig_AB);
                for eig_ = eig_AB
                    abs_eig_ = abs(eig_);
                    if abs_eig_ > 0.5
                        err_1 = [err_1, eig_];
                    else
                        err_0 = [err_0, eig_];
                    end
                end
                errAB = max([abs(err_1 - 1), abs(err_0)]);
                % fprintf("Projector AB error: %.9e\n", errAB);

                if (errAB < 1e-4)
                    break
                elseif new_chi_AB == 1
                    break
                else
                    new_chi_AB = new_chi_AB - 1;
                end
            end
            
            
            while true

                % SBA_cut = diag(SBA);
                SBA_cut = sqrt(SBA_cut(length(SBA_cut) - new_chi_BA + 1:length(SBA_cut)));

                % % fprintf("bond BA SVD error: %.9e\n", 1 - sum(SBA_cut(1:new_chi_BA)) / sum(SBA_cut));
                % SBA_cut = 1.0 ./ sqrt(SBA_cut(1:new_chi_BA));
                % % fprintf('Building Projectors for bond BA\n');

                VBAh = VBA';
                UBAh = UBA';
                
                diag_sqrt_inv_SBA_cut = diag(SBA_cut);

                % PBA_left = RBA_up_right * VBA(:, 1:new_chi_BA) * diag_sqrt_inv_SBA_cut;
                % PBA_right = transpose(diag_sqrt_inv_SBA_cut * UBAh(1:new_chi_BA, :) * RBA_up_left);
                PBA_left = RBA_up_right * VBA(:, length(SBA_cut) - new_chi_BA + 1:length(SBA_cut)) * diag_sqrt_inv_SBA_cut;
                PBA_right = transpose(diag_sqrt_inv_SBA_cut * UBAh(length(SBA_cut) - new_chi_BA + 1:length(SBA_cut), :) * RBA_up_left);
                % PBA_right = transpose(pinv(PBA_left));
                test_mat = PBA_left * transpose(PBA_right);
                eig_BA = eig(test_mat);
                % fprintf("CTM error: %.9e\n", trace(RBA_0 - RBA_up_left * PBA_left * transpose(PBA_right) * RBA_up_right) / trace(RBA_0))
                err_1 = [];
                err_0 = [];
                eig_BA = transpose(eig_BA);
                for eig_ = eig_BA
                    abs_eig = abs(eig_);
                    if abs_eig > 0.5
                        err_1 = [err_1, eig_];
                    else
                        err_0 = [err_0, eig_];
                    end
                end
                
                errBA = max([abs(err_1 - 1), abs(err_0)]);
                % fprintf("Projector BA error: %.9e\n", errBA);

                if (errBA < 1e-4)
                    break
                elseif new_chi_BA == 1
                    break
                else
                    new_chi_BA = new_chi_BA - 1;
                end

            end
            
            PAB_left = RESHAPE(PAB_left, ...
                [size(obj.TA{LIO(1)}, 4), size(obj.a, LIO(3)), size(obj_low.a, LIO(3)),  new_chi_AB]);

            PAB_right = RESHAPE(PAB_right, ...
                [size(obj.TB{LIO(1)}, 1), size(obj.b, LIO(1)), size(obj_low.b, LIO(1)),  new_chi_AB]);

            % fprintf("Projectors for bond AB : Built, new_chi_AB: %d\n", new_chi_AB)

            PBA_left = RESHAPE(PBA_left, ...
                [size(obj.TB{LIO(1)}, 4), size(obj.b, LIO(3)), size(obj_low.b, LIO(3)), new_chi_BA]);
            PBA_right = RESHAPE(PBA_right, ...
                [size(obj.TA{LIO(1)}, 1), size(obj.a, LIO(1)), size(obj_low.a, LIO(1)), new_chi_BA]);

            % fprintf("Projectors for bond BA: Built, new_chi_BA: %d\n", new_chi_BA)

            CB_temp_0 = obj.CTMCornerUpdate(obj.CA{LIO(1)}, obj.TA{LIO(4)}, PBA_left, 0);
            CA_temp_0 = obj.CTMCornerUpdate(obj.CB{LIO(1)}, obj.TB{LIO(4)}, PAB_left, 0);

            CB_temp_1 = obj.CTMCornerUpdate(obj.CA{LIO(2)}, obj.TA{LIO(2)}, PAB_right, 1);
            CA_temp_1 = obj.CTMCornerUpdate(obj.CB{LIO(2)}, obj.TB{LIO(2)}, PBA_right, 1);
    
            TB_temp = obj.CTMEdgeUpdate(PBA_right, obj.TA{LIO(1)}, permute(obj.a, LIO), conj(permute(obj_low.a, LIO)), PAB_left);
            TA_temp = obj.CTMEdgeUpdate(PAB_right, obj.TB{LIO(1)}, permute(obj.b, LIO), conj(permute(obj_low.b, LIO)), PBA_left);

            if update_now ~= 1
                return
            end

            if normalize == 1
                obj.CA{LIO(1)} = CA_temp_0 / norm(CA_temp_0(:));
                obj.CB{LIO(1)} = CB_temp_0 / norm(CB_temp_0(:));

                obj.CA{LIO(2)} = CA_temp_1 / norm(CA_temp_1(:));
                obj.CB{LIO(2)} = CB_temp_1 / norm(CB_temp_1(:));

                obj.TA{LIO(1)} = TA_temp / norm(TA_temp(:));
                obj.TB{LIO(1)} = TB_temp / norm(TB_temp(:));
            else
                obj.CA{LIO(1)} = CA_temp_0;
                obj.CB{LIO(1)} = CB_temp_0;

                obj.CA{LIO(2)} = CA_temp_1;
                obj.CB{LIO(2)} = CB_temp_1;

                obj.TA{LIO(1)} = TA_temp;
                obj.TB{LIO(1)} = TB_temp;
            end
        end

        function CTMRenorm(obj, obj_low, bond_dir, svd_tol, pinv_tol, normalize, update_now)

            LIO = [mod([0, 1, 2, 3] + bond_dir, 4) + 1, 5];
            RIO = [mod([1, 2, 3, 0] + bond_dir, 4) + 1, 5];


            leftAB = obj.CTMContraction(obj.a, conj(obj_low.a), obj.TA, obj.CA, LIO, 0);
            rightAB = obj.CTMContraction(obj.b, conj(obj_low.b), obj.TB, obj.CB, RIO, 0);
            [U, S, V] = svd(leftAB * rightAB, "econ");
            sqrt_S_cut = diag(S);
            new_chi_AB = max(1, min(sum(sqrt_S_cut > svd_tol), obj.chimax));
            

            leftBA = obj.CTMContraction(obj.b, conj(obj_low.b), obj.TB, obj.CB, LIO, 0);
            rightBA = obj.CTMContraction(obj.a, conj(obj_low.a), obj.TA, obj.CA, RIO, 0);
            [U_, S_, V_] = svd(leftBA * rightBA, "econ");
            sqrt_S__cut = diag(S_);
            new_chi_BA = max(1, min(sum(sqrt_S__cut > svd_tol), obj.chimax));

            new_chi = max(new_chi_BA, new_chi_AB);
            new_chi_AB = new_chi;
            new_chi_BA = new_chi;
            % fprintf("New chi: %d\n", new_chi);

            if new_chi > length(S)
                sqrt_S_cut = cat(2, sqrt_S_cut, zeros(1, new_chi - length(sqrt_S_cut)));
            end
            
            if new_chi > length(S_)
                sqrt_S__cut = cat(2, sqrt_S__cut, zeros(1, new_chi - length(sqrt_S__cut)));
            end


            % fprintf("bond AB SVD error: %.9e\n", 1 - sum(sqrt_S_cut(1:new_chi_AB)) / sum(sqrt_S_cut));
            % fprintf('Building Projectors for bond AB\n');

            Vh = V';
            Uh = U';
            sqrt_S_cut = diag(sqrt(sqrt_S_cut(1:new_chi_AB)));

            % PAB_left = pinv(leftAB, pinv_tol) * U(:, 1:new_chi_AB) * sqrt_S_cut;
            % PAB_right = pinv(transpose(rightAB), pinv_tol) * transpose(Vh(1:new_chi_AB,:)) * sqrt_S_cut;
            PAB_left = rightAB * V(:, 1:new_chi_AB) * pinv(sqrt_S_cut);
            PAB_right = transpose(pinv(sqrt_S_cut) * Uh(1:new_chi_AB, :) * leftAB);

            eig_AB = eig(PAB_left * transpose(PAB_right));
                % % fprintf("AB eigenvalues\n");
                % % fprintf("Real part\n")
                % % fprintf("%.9e\t", real(eig_AB));
                % % fprintf("\n");
                % % fprintf("Imaginary part\n")
                % % fprintf("%.9e\t", imag(eig_AB));
                % % fprintf("\n")
            err_1 = [];
            err_0 = [];
            eig_AB = transpose(eig_AB);
            for eig_ = eig_AB
                abs_eig_ = abs(eig_);
                if abs_eig_ > 0.5
                    err_1 = [err_1, eig_];
                else
                    err_0 = [err_0, eig_];
                end
            end
            errAB = max([abs(err_1 - 1), abs(err_0)]);
            % fprintf("Projector AB error: %.9e\n", errAB);

            
            % PAB_left = kron(PAB_left, conj(PAB_left));
            PAB_left = RESHAPE(PAB_left, ...
                [size(obj.TA{LIO(1)}, 4), size(obj.a, LIO(3)), size(obj_low.a, LIO(3)),  new_chi_AB]);

            % PAB_right = kron(PAB_right, conj(PAB_right));
            PAB_right = RESHAPE(PAB_right, ...
                [size(obj.TB{LIO(1)}, 1), size(obj.b, LIO(1)), size(obj_low.b, LIO(1)),  new_chi_AB]);

            
            Vh_ = V_';
            Uh_ = U_';
            % fprintf("bond BA SVD error: %.9e\n", 1 - sum(sqrt_S__cut(1:new_chi_BA)) / sum(sqrt_S__cut));
            % fprintf('Building Projectors for bond BA\n');

            sqrt_S__cut = diag(sqrt(sqrt_S__cut(1:new_chi_BA)));

            % PBA_left = pinv(leftBA, pinv_tol) * U_(:, 1:new_chi_BA) * sqrt_S__cut;
            % PBA_right = pinv(transpose(rightBA), pinv_tol) * transpose(Vh_(1:new_chi_BA,:)) * sqrt_S__cut;
            PBA_left = rightBA * V_(:, 1:new_chi_BA) * pinv(sqrt_S__cut);
            PBA_right = transpose(pinv(sqrt_S__cut) * Uh_(1:new_chi_BA, :) * leftBA);

            eig_BA = eig(PBA_left * transpose(PBA_right));
                % % fprintf("AB eigenvalues\n");
                % % fprintf("Real part\n")
                % % fprintf("%.9e\t", real(eig_AB));
                % % fprintf("\n");
                % % fprintf("Imaginary part\n")
                % % fprintf("%.9e\t", imag(eig_AB));
                % % fprintf("\n")
            err_1 = [];
            err_0 = [];
            eig_BA = transpose(eig_BA);
            for eig_ = eig_BA
                abs_eig_ = abs(eig_);
                if abs_eig_ > 0.5
                    err_1 = [err_1, eig_];
                else
                    err_0 = [err_0, eig_];
                end
            end
            errBA = max([abs(err_1 - 1), abs(err_0)]);
            % fprintf("Projector BA error: %.9e\n", errBA);

            % PBA_left = kron(PBA_left, conj(PBA_left));
            PBA_left = RESHAPE(PBA_left, [size(obj.TB{LIO(1)}, 4), size(obj.b, LIO(3)), size(obj_low.b, LIO(3)), new_chi_BA]);

            % PBA_right = kron(PBA_right, conj(PBA_right));
            PBA_right = RESHAPE(PBA_right, [size(obj.TA{LIO(1)}, 1), size(obj.a, LIO(1)), size(obj_low.a, LIO(1)), new_chi_BA]);

            % fprintf("Projectors for bond BA: Built, new_chi_BA: %d\n", new_chi_BA)

            CB_temp_0 = obj.CTMCornerUpdate(obj.CA{LIO(1)}, obj.TA{LIO(4)}, PBA_left, 0);
            CA_temp_0 = obj.CTMCornerUpdate(obj.CB{LIO(1)}, obj.TB{LIO(4)}, PAB_left, 0);

            CB_temp_1 = obj.CTMCornerUpdate(obj.CA{LIO(2)}, obj.TA{LIO(2)}, PAB_right, 1);
            CA_temp_1 = obj.CTMCornerUpdate(obj.CB{LIO(2)}, obj.TB{LIO(2)}, PBA_right, 1);
    
            TB_temp = obj.CTMEdgeUpdate(PBA_right, obj.TA{LIO(1)}, permute(obj.a, LIO), conj(permute(obj_low.a, LIO)), PAB_left);
            TA_temp = obj.CTMEdgeUpdate(PAB_right, obj.TB{LIO(1)}, permute(obj.b, LIO), conj(permute(obj_low.b, LIO)), PBA_left);

            if update_now ~= 1
                return
            end

            if normalize == 1
                obj.CA{LIO(1)} = CA_temp_0 / norm(CA_temp_0(:));
                obj.CB{LIO(1)} = CB_temp_0 / norm(CB_temp_0(:));

                obj.CA{LIO(2)} = CA_temp_1 / norm(CA_temp_1(:));
                obj.CB{LIO(2)} = CB_temp_1 / norm(CB_temp_1(:));

                obj.TA{LIO(1)} = TA_temp / norm(TA_temp(:));
                obj.TB{LIO(1)} = TB_temp / norm(TB_temp(:));
            else
                obj.CA{LIO(1)} = CA_temp_0;
                obj.CB{LIO(1)} = CB_temp_0;

                obj.CA{LIO(2)} = CA_temp_1;
                obj.CB{LIO(2)} = CB_temp_1;

                obj.TA{LIO(1)} = TA_temp;
                obj.TB{LIO(1)} = TB_temp;
            end
        end

        function [U_left, U_right] = HamiltonianExp(~, H, tau, it_rt)
            if it_rt == 0
                u = expm(-tau * H);
                u = u / trace(u);
            elseif it_rt == 1
                u = expm(-1j * tau * H);
            end

            u = RESHAPE(permute(RESHAPE(u, [2, 2, 2, 2]), [1, 3, 2, 4]), [4, 4]);

            [U, S, V] = svd(u);
            Vh = V';
            U_left = U * sqrt(S);
            U_left = RESHAPE(U_left, [2, 2, 4]);
            U_left = permute(U_left, [2, 3, 1]);
            
            U_right = sqrt(S) * Vh;
            U_right = RESHAPE(U_right, [4, 2, 2]);
            U_right = permute(U_right, [3, 1, 2]);
        end
        
        function [U_left, U_right] = HamiltonianExp_Ising(~, tau, it_rt)
            Id = eye(2, 2);
            Sz = [[1, 0]; [0, -1]];

            if tau == 0 % This is just an identity for a general truncation
                U_left = eye(2, 2);
                U_left = permute(U_left, [2, 3, 1]); 
                U_right = U_left;
                return
            end

            % if it_rt == 0
            %     u = cos(-tau * 1j) * Id - 1j * sin(-tau * 1j) * H;
            % else
            %     u = cos(tau) * Id - 1j * sin(tau) * H;
            % end

            if it_rt == 0
                coeff_1 = sqrt(cos(-1j * tau));
                coeff_2 = sqrt(1j * sin(-1j * tau));
            else
                coeff_1 = sqrt(cos(tau));
                coeff_2 = sqrt(1j * sin(tau));
            end

            U_left = zeros(2, 2, 2);
            U_left(:, :, 1) = coeff_1 * Id;
            U_left(:, :, 2) = coeff_2 * Sz;

            U_left = permute(U_left, [2, 3, 1]); 

            %  1 (contract with ket)
            %  |
            %  u_left--2 
            %  |
            %  3 (contract with bra)
            
            U_right = U_left;

            % % disp(u)

            % u = RESHAPE(permute(RESHAPE(u, [2, 2, 2, 2]), [1, 3, 2, 4]), [4, 4]);
            % % disp(u)
            
            % [U, S, V] = svd(u);
            % Vh = V';
            % % fprintf("%.9e ", diag(S))
            % % fprintf("\n")
            % U_left = U(:,1:2) * sqrt(S(1:2, 1:2));
            % U_left = RESHAPE(U_left, [2, 2, 2]);
            % U_left = permute(U_left, [2, 3, 1]);
            
            % U_right = sqrt(S(1:2, 1:2)) * Vh(1:2, :);
            % U_right = RESHAPE(U_right, [2, 2, 2]);
            % U_right = permute(U_right, [3, 1, 2]);
        end

        function [expvalue, rhoAB] = ExpectationValue(obj, obj_low, O, bond_dir, normalize)
            LIO = [mod([0, 1, 2, 3] + bond_dir, 4) + 1, 5];
            RIO = [mod([1, 2, 3, 0] + bond_dir, 4) + 1, 5];

            leftAB = obj.CTMContraction(obj.a, conj(obj_low.a), obj.TA, obj.CA, LIO, 1);
            rightAB = obj.CTMContraction(obj.b, conj(obj_low.b), obj.TB, obj.CB, RIO, 1);


            temp = tensorprod(leftAB, rightAB, [4, 5, 6], [1, 2, 3], NumDimensionsA=8);
            temp = tensorprod(obj.CA{LIO(4)}, temp, 2, 1, NumDimensionsA=2);
            temp = tensorprod(obj.TA{LIO(3)}, temp, [2, 3, 4], [2, 3, 1], NumDimensionsA=4);
            temp = tensorprod(temp, obj.TB{LIO(3)}, [1, 5, 6], [4, 2, 3], NumDimensionsA=8);
            rhoAB = tensorprod(temp, obj.CB{LIO(3)}, [3, 6], [1, 2], NumDimensionsA=6);
            rhoAB = permute(rhoAB, [1, 3, 2, 4]);
            rhoAB = RESHAPE(rhoAB, [obj.d * obj.d, obj.d * obj.d]);
            if normalize == 1
                [eigvec, eigval] = eig(rhoAB, "vector");
                % rhoAB = eigvec * diag(eigval ./ sign(eigval)) * eigvec';
                % % fprintf("%.9e ", eigval);
                % % fprintf("\n")
                rhoAB = rhoAB / trace(rhoAB);
            end
            if O == -1
                expvalue = 0;
            else
                expvalue = trace(rhoAB * O);
            end
        end
        
        function [expvalue, rhoAB] = ExpectationValue_2(obj, obj_low, O, bond_dir, normalize)
            LIO = [mod([0, 1, 2, 3] + bond_dir, 4) + 1, 5];
            RIO = [mod([1, 2, 3, 0] + bond_dir, 4) + 1, 5];

            leftAB = obj.CTMContraction(obj.a, conj(obj_low.a), obj.TA, obj.CA, LIO, 1);
            rightAB = obj.CTMContraction(obj.b, conj(obj_low.b), obj.TB, obj.CB, RIO, 1);


            temp = tensorprod(leftAB, rightAB, [4, 5, 6], [1, 2, 3], NumDimensionsA=8);
            temp = tensorprod(obj.CA{LIO(4)}, temp, 2, 1, NumDimensionsA=2);
            temp = tensorprod(obj.TA{LIO(3)}, temp, [2, 3, 4], [2, 3, 1], NumDimensionsA=4);
            temp = tensorprod(temp, obj.TB{LIO(3)}, [1, 5, 6], [4, 2, 3], NumDimensionsA=8);
            rhoAB = tensorprod(temp, obj.CB{LIO(3)}, [3, 6], [1, 2], NumDimensionsA=6);
            rhoAB = permute(rhoAB, [1, 3, 2, 4]);
            rhoAB = RESHAPE(rhoAB, [obj.d * obj.d, obj.d * obj.d]);
            if normalize == 1
                [eigvec, eigval] = eig(rhoAB, "vector");
                % rhoAB = eigvec * diag(eigval ./ sign(eigval)) * eigvec';
                % % fprintf("%.9e ", eigval);
                % % fprintf("\n")
                rhoAB = rhoAB / trace(rhoAB);
            end
            if O == -1
                expvalue = 0;
            else
                expvalue = trace(rhoAB * O);
            end
        end

        function t = Overlap(obj, obj_low, ctm_tol, svd_tol, pinv_tol)
            
            error = inf;
            iter = 0;
            t_old = 0;
            t = 0;

            obj.CTMInitialize(obj_low, "eye", 1);


            while (error > ctm_tol) && (iter < 400)
                
                for jj = 1:2
                    for ii = 0:3
                        obj.CTMRenorm(obj_low, ii, svd_tol, pinv_tol, 1, 1);
                    end
                end

                a1 = trace(obj.CA{1} * obj.CB{2} * obj.CA{3} * obj.CB{4});
                
                LIO = [1, 2, 3, 4, 5];
                RIO = [2, 3, 4, 1, 5];
                RIO_ = [3, 4, 1, 2, 5];
                LIO_ = [4, 1, 2, 3, 5];

                ul = obj.CTMContraction(obj.a, conj(obj_low.a), obj.TA, obj.CA, LIO, 0);
                ur = obj.CTMContraction(obj.b, conj(obj_low.b), obj.TB, obj.CB, RIO, 0);
                bl = obj.CTMContraction(obj.a, conj(obj_low.a), obj.TA, obj.CA, LIO_, 0);
                br = obj.CTMContraction(obj.b, conj(obj_low.b), obj.TB, obj.CB, RIO_, 0);

                a2 = trace(ul * ur * br * bl);

                b1_ul = tensorprod(obj.CA{1}, obj.TA{1}, 2, 1, NumDimensionsA=2);
                b1_ur = tensorprod(obj.TB{1}, obj.CB{2}, 4, 1, NumDimensionsA=4);
                b1_br = tensorprod(obj.CA{3}, obj.TA{3}, 2, 1, NumDimensionsA=2);
                b1_bl = tensorprod(obj.TB{3}, obj.CB{4}, 4, 1, NumDimensionsA=4);
                b1_l = tensorprod(b1_ul, b1_bl, [1, 2, 3], [4, 2, 3], NumDimensionsA=4);
                b1_r = tensorprod(b1_ur, b1_br, [2, 3, 4], [2, 3, 1], NumDimensionsA=4);
                b1 = tensorprod(b1_l, b1_r, [1, 2], [1, 2], NumDimensionsA=2);

                b2_ul = tensorprod(obj.TA{4}, obj.CA{1}, 4, 1, NumDimensionsA=4);
                b2_ur = tensorprod(obj.CB{2}, obj.TB{2}, 2, 1, NumDimensionsA=2);
                b2_br = tensorprod(obj.TA{2}, obj.CA{3}, 4, 1, NumDimensionsA=4);
                b2_bl = tensorprod(obj.CB{4}, obj.TB{4}, 2, 1, NumDimensionsA=2);
                b2_u = tensorprod(b2_ul, b2_ur, [2, 3, 4], [2, 3, 1], NumDimensionsA=4);
                b2_b = tensorprod(b2_bl, b2_br, [1, 2, 3], [4, 2, 3], NumDimensionsA=4);
                b2 = tensorprod(b2_u, b2_b, [1, 2], [1, 2], NumDimensionsA=2);

                t = (a1 * a2) / (b1 * b2);
                % fprintf("abs(Overlap): %.9e\n", abs(t));
                error = abs(abs(t) - abs(t_old));
                % fprintf("error: %.9e\n", error);
                t_old = t;
            end
            % fprintf("%.9e, %.9e, %.9e, %.9e\n", a1, a2, b1, b2);
            % a2 = obj.Energy(obj_low, eye(obj.d * obj.d));
            % t = a2;
        end
        
        function rho = rhoAB(obj, obj_low, normalize)
            rho = 0;
            for bond_dir = 0:3
                [~, temp_rho] = obj.ExpectationValue(obj_low, -1, bond_dir, normalize);
                rho = rho + temp_rho;
            end
            rho = rho / 4.0;
        end
        
        function ApplyLocalGate(obj, gate_local, whichone)

            if whichone == 0
                obj.a = tensorprod(obj.a, gate_local, 5, 1, NumDimensionsA=5);
                % obj.a = obj.a / norm(obj.a(:));
            elseif whichone == 1
                obj.b = tensorprod(obj.b, gate_local, 5, 1, NumDimensionsA=5);
                % obj.b = obj.b / norm(obj.b(:));
            end
        end

        function ntu_error_min = ApplyGate(obj, gate_a, gate_b, bond_dir, method, ntu_tol, svd_tol, pinv_tol)

            indices_order = [mod([0, 1, 2, 3] + bond_dir, 4) + 1, 5];
            indices_order_ = [mod([0, 1, 2, 3] - bond_dir, 4) + 1, 5];
            ntu_error = 0;
            
            temp_A = tensorprod(permute(obj.a, indices_order), permute(gate_a, [1, 3, 2]), 5, 1, NumDimensionsA=5);
        
            temp_B = tensorprod(permute(obj.b, indices_order), permute(gate_b, [1, 3, 2]), 5, 1, NumDimensionsA=5);

            if method == "Notrunc"
                % fprintf("No truncation\n")
                temp_A = permute(temp_A, [1, 2, 3, 6, 4, 5]);
                temp_B = permute(temp_B, [1, 6, 2, 3, 4, 5]);
                a_ = RESHAPE(temp_A, [size(temp_A, 1), size(temp_A, 2), size(temp_A, 3) * size(temp_A, 4), size(temp_A, 5), size(temp_A, 6)]);
                b_ = RESHAPE(temp_B, [size(temp_B, 1) * size(temp_B, 2), size(temp_B, 3), size(temp_B, 4), size(temp_B, 5), size(temp_B, 6)]);

                obj.a = permute(a_, indices_order_);
                obj.b = permute(b_, indices_order_);
    
                obj.a = obj.a / norm(obj.a(:));
                obj.b = obj.b / norm(obj.b(:));
                return;
            end

            % fprintf("NTU\n")

            temp_A = permute(temp_A, [1, 2, 4, 5, 3, 6]);
            temp_B = permute(temp_B, [2, 3, 4, 5, 1, 6]);

            [QA, RA] = qr(RESHAPE(temp_A, [size(temp_A, 1) * size(temp_A, 2) * ...
                size(temp_A, 3) * size(temp_A, 4), ...
                size(temp_A, 5) * size(temp_A, 6)]), "econ");
            
            QA = RESHAPE(QA, [size(temp_A, 1), size(temp_A, 2), ...
                size(temp_A, 3), size(temp_A, 4), size(QA, 2)]);
            QA = permute(QA, [1, 2, 5, 3, 4]);


            [QB, RB] = qr(RESHAPE(temp_B, [size(temp_B, 1) * size(temp_B, 2) ...
                * size(temp_B, 3) * size(temp_B, 4), ...
                size(temp_B, 5) * size(temp_B, 6)]), "econ");
            QB = RESHAPE(QB, [size(temp_B, 1), size(temp_B, 2), ...
                size(temp_B, 3), size(temp_B, 4), size(QB, 2)]);
            QB = permute(QB, [5, 1, 2, 3, 4]);

            RB = transpose(RB);

            RARB = RA * RB;
            [UA, S, UB] = svd(RARB);
            UB = UB';
            S_diag = diag(S);
            % % disp(S_diag)
            % newD = min(obj.D, sum(S_diag > svd_tol));
            % fprintf("SVD tol:%.9e\n", svd_tol)
            newD = min(obj.D, sum(S_diag > svd_tol));
            
            % % fprintf('Apply Gate SVD error: %.9e\n', 1 - sum(S_diag(1:newD)) / sum(S_diag));

            % [UA, S, UB] = eig(RARB);
            % diff_U = UA - UB;
            % % fprintf("UAUB error %.9e\n", max(abs(diff_U(:))))
            % UB = UB';
            % S_diag = diag(S);
            % % newD = min(obj.D, sum(S_diag > svd_tol));
            % newD = min(obj.D, sum(abs(S_diag) > svd_tol));
            
            % fprintf('Apply Gate SVD error: %.9e\n', 1 - sum(abs(S_diag(1:newD))) / sum(abs(S_diag)));

            % fprintf('New bond dimension: %d\n', newD);


            RARB = RESHAPE(RARB, [numel(RARB), 1]);

            

            if method == "NTU"

                temp_A = permute(obj.a, indices_order);
                temp_B = permute(obj.b, indices_order);
                
                temp_1 = tensorprod(temp_B, conj(temp_B), [1, 2, 5], [1, 2, 5], NumDimensionsA=5);
                temp_2 = tensorprod(temp_A, conj(temp_A), [2, 3, 5], [2, 3, 5], NumDimensionsA=5);

                temp_3 = tensorprod(temp_B, conj(temp_B), [1, 2, 4, 5], [1, 2, 4, 5], NumDimensionsA=5);
                temp_3 = tensorprod(temp_3, QA, 1, 1, NumDimensionsA=2);
                temp_3 = tensorprod(temp_3, conj(QA), [1, 5], [1, 5], NumDimensionsA=5);
                
                temp_4 = tensorprod(temp_A, conj(temp_A), [2, 3, 4, 5], [2, 3, 4 5], NumDimensionsA=5);
                temp_4 = tensorprod(temp_4, QB, 1, 3, NumDimensionsA=2);
                temp_4 = tensorprod(temp_4, conj(QB), [1, 5], [3, 5], NumDimensionsA=5);

                temp_5 = tensorprod(temp_B, conj(temp_B), [1, 4, 5], [1, 4, 5], NumDimensionsA=5);
                temp_6 = tensorprod(temp_A, conj(temp_A), [3, 4, 5], [3, 4, 5], NumDimensionsA=5);

                

                g_left = tensorprod(temp_1, temp_2, [1, 3], [1, 3], NumDimensionsA=4);
        
                g_left = tensorprod(g_left, temp_3, [1, 2], [1, 4], NumDimensionsA=4);
                g_left = permute(g_left, [1, 2, 4, 6, 3, 5]);

                g_right = tensorprod(temp_5, temp_6, [2, 4], [1, 3], NumDimensionsA=4);
                g_right = tensorprod(g_right, temp_4, [3, 4], [3, 6], NumDimensionsA=4);
                g_right = permute(g_right, [4, 6, 1, 2, 3, 5]);


                g = tensorprod(g_left, g_right, [1, 2, 3, 4], [1, 2, 3, 4], NumDimensionsA=6);
                % test_g = RESHAPE(permute(g, [1, 3, 2, 4]), [size(g, 1) * size(g, 3), size(g, 2) * size(g, 4)]);
                % % disp(eig(test_g))

                J = tensorprod(RESHAPE(permute(g, [2, 4, 1, 3]), ...
                        [size(g, 2), size(g, 4), size(g, 1) * size(g, 3)]), ...
                        RARB, 3, 1, NumDimensionsA=4);

                F0 = tensorprod(RESHAPE(J, [numel(J), 1]), conj(RARB), [1, 2], [1, 2], NumDimensionsA=2);

                ntu_error_min = inf;
                MA_min = 0;
                MB_min = 0;

                for newD_ = newD:-1:1
                    % fprintf("Current newD: %d\n", newD_);
                    MA = UA(:,1:newD_) * diag(sqrt(S_diag(1:newD_)));
                    MB = diag(sqrt(S_diag(1:newD_))) * UB(1:newD_,:);
                    % MA = UA(:,1:newD) * diag(sqrt(S_diag(1:newD)));
                    % MB = diag(sqrt(S_diag(1:newD))) * UB(1:newD,:);
                    % MA = RA;
                    % MB = RB;
                    % % disp(size(MA))
                    % % disp(size(MB))
                    ntu_error = inf;
                    iter = 0;
                    ntu_error_old = inf;
                    while (abs(ntu_error) > ntu_tol) && iter < 100
                        
                        iter = iter + 1;

                        gA = tensorprod(tensorprod(g, MB, 3, 2, NumDimensionsA=4), ...
                            conj(MB), 3, 2, NumDimensionsA=4);
                        % test_gA = RESHAPE(permute(gA, [1, 3, 2, 4]), [size(gA, 1) * size(gA, 3), size(gA, 2) * size(gA, 4)]);
                        % % disp(eig(test_gA))
                        gA = permute(gA, [2, 4, 1, 3]);
                        JA = tensorprod(J, conj(MB), 2, 2, NumDimensionsA=2);

                        gA = RESHAPE(gA, [size(g, 2) * size(MB, 1), ...
                            size(g, 1) * size(MB, 1)]);
                        
                        JA = RESHAPE(JA, [size(g, 2) * size(MB, 1), 1]);
                        MA = pinv(gA, 1e-15) * JA;
                        % MA = MA / norm(MA(:));
                        MA = RESHAPE(MA, [size(g, 1), size(MB, 1)]);

                        gB = tensorprod(tensorprod(g, MA, 1, 1, NumDimensionsA=4), conj(MA), 1, 1, NumDimensionsA=4);
                        gB = permute(gB, [4, 2, 3, 1]);
                        JB = tensorprod(conj(MA), J, 1, 1, NumDimensionsA=2);
                        gB = RESHAPE(gB, [size(MA, 2) * size(g, 4), ...
                            size(MA, 2) * size(g, 3)]);
                        JB = RESHAPE(JB, [size(MA, 2) * size(g, 4), 1]);
                        MB = pinv(gB, 1e-15) * JB;
                        % MB = MB / norm(MB(:));
                        MB = RESHAPE(MB, [size(MA, 2), size(g, 3)]);

                        MAMB = MA * MB;
                        delta = RESHAPE(MAMB, [size(MA, 1) * size(MB, 2), 1]) - RARB;
                        F = tensorprod(RESHAPE(permute(g, [1, 3, 2, 4]), ...
                            [size(g, 1) * size(g, 3), size(g, 2) * size(g, 4), ]), ...
                        delta, 1, 1, NumDimensionsA=2);
                        F = tensorprod(conj(delta), F, [1, 2], [1, 2], NumDimensionsA=2);
                        ntu_error = F / F0;
                        if (abs((ntu_error - ntu_error_old) / ntu_error_old) < 1e-9)
                            break;
                        end
                        ntu_error_old = ntu_error;
                    end
                    % fprintf("bond_dir: %d, NTU error: %.9e, number_of_iteration: %d\n", bond_dir, ntu_error, iter);
                    if abs(ntu_error) <= abs(ntu_error_min)
                        MA_min = MA;
                        MB_min = MB;
                        ntu_error_min = ntu_error;
                    end
                    % fprintf("Optimized NTU error: %.9e\n", ntu_error_min);
                end

                MA = MA_min;
                MB = MB_min;

                a_ = permute(tensorprod(QA, MA, 3, 1, NumDimensionsA=5), [1, 2, 5, 3, 4]);
                b_ = tensorprod(MB, QB, 2, 1, NumDimensionsA=2);

            end




            obj.a = permute(a_, indices_order_);
            obj.b = permute(b_, indices_order_);

            obj.a = obj.a / norm(obj.a(:));
            obj.b = obj.b / norm(obj.b(:));

        end

        function temp = FindFixedPoint_TT_fun(~, AU, BD, C)
            c = RESHAPE(C, [size(BD, 4), size(AU, 1)]);
            temp = tensorprod(c, AU, 2, 1, NumDimensionsA=2);
            temp = tensorprod(BD, temp, [4, 2, 3], [1, 2, 3], NumDimensionsA=4);
            temp = RESHAPE(temp, [size(BD, 4) * size(AU, 1), 1]);
        end

        function C = FindFixedPoint_TT(obj, AU, BD)
            fun = @(x)obj.FindFixedPoint_TT_fun(AU, BD, x);
            [C, ~] = eigs(fun, size(BD, 4) * size(AU, 1), 1, 'largestabs', 'Tolerance',1e-20,'MaxIterations',1000,'Display',0);
            C = RESHAPE(C, [size(BD, 4), size(AU, 1)]);
        end

        function temp = FindFixedPointLeftFun(~, T1, T2, T3, T4, a, b, X)
            
            x = RESHAPE(X, [size(T3, 4), size(a, 1), size(a, 1), size(T1, 1)]);

            temp = tensorprod(conj(T3), x, 4, 1, NumDimensionsA=4);
            temp = tensorprod(temp, a, [2, 4], [4, 1], NumDimensionsA=6);
            temp = tensorprod(temp, conj(a), [2, 3, 7], [4, 1, 5], NumDimensionsA=7);
            temp = tensorprod(temp, T1, [2, 3, 5], [1, 2, 3], NumDimensionsA=6);

            temp = tensorprod(conj(T4), temp, 4, 1, NumDimensionsA=4);
            temp = tensorprod(temp, b, [2, 4], [4, 1], NumDimensionsA=6);
            temp = tensorprod(temp, conj(b), [2, 3, 7], [4, 1, 5], NumDimensionsA=7);
            temp = tensorprod(temp, T2, [2, 3, 5], [1, 2, 3], NumDimensionsA=6);

            % % fprintf("Matrix multiplication done\n")

            temp = RESHAPE(temp, [size(T3, 4) * size(a, 1) * size(a, 1) * size(T1, 1), 1]);
        end

        function temp = FindFixedPointRightFun(~, T1, T2, T3, T4, a, b, X)
            
            x = RESHAPE(X, [size(T2, 4), size(a, 1), size(a, 1), size(T4, 1)]);

            temp = tensorprod(T2, x, 4, 1, NumDimensionsA=4);
            temp = tensorprod(temp, b, [2, 4], [2, 3], NumDimensionsA=6);
            temp = tensorprod(temp, conj(b), [2, 3, 7], [2, 3, 5], NumDimensionsA=7);
            temp = tensorprod(temp, conj(T4), [2, 4, 6], [1, 2, 3], NumDimensionsA=6);

            temp = tensorprod(T1, temp, 4, 1, NumDimensionsA=4);
            temp = tensorprod(temp, a, [2, 4], [2, 3], NumDimensionsA=6);
            temp = tensorprod(temp, conj(a), [2, 3, 7], [2, 3, 5], NumDimensionsA=7);
            temp = tensorprod(temp, conj(T3), [2, 4, 6], [1, 2, 3], NumDimensionsA=6);

            % % fprintf("Matrix multiplication done\n")

            temp = RESHAPE(temp, [size(T2, 4) * size(b, 3) * size(b, 3) * size(T4, 1), 1]);
        end

        function temp = FindFixedPointLeftBoundaryFun(~, T1, T2, T3, T4, X)
            
            x = RESHAPE(X, [size(T3, 4), size(T1, 1)]);

            temp = tensorprod(conj(T3), x, 4, 1, NumDimensionsA=4);
            temp = tensorprod(temp, T1, [2, 3, 4], [1, 2, 3], NumDimensionsA=4);

            temp = tensorprod(conj(T4), temp, 4, 1, NumDimensionsA=4);
            temp = tensorprod(temp, T2, [2, 3, 4], [1, 2, 3], NumDimensionsA=4);

            % % fprintf("Matrix multiplication done\n")

            temp = RESHAPE(temp, [size(T3, 4) * size(T1, 1), 1]);
        end

        function temp = FindFixedPointRightBoundaryFun(~, T1, T2, T3, T4, X)
            
            x = RESHAPE(X, [size(T2, 4), size(T4, 1)]);

            temp = tensorprod(T2, x, 4, 1, NumDimensionsA=4);
            temp = tensorprod(temp, conj(T4), [2, 3, 4], [1, 2, 3], NumDimensionsA=4);

            temp = tensorprod(T1, temp, 4, 1, NumDimensionsA=4);
            temp = tensorprod(temp, conj(T3), [2, 3, 4], [1, 2, 3], NumDimensionsA=4);

            % % fprintf("Matrix multiplication done\n")

            temp = RESHAPE(temp, [size(T2, 4) * size(b, 3) * size(b, 3) * size(T4, 1), 1]);
        end

        function temp = FindFixedPointLeft1Fun(~, T1, T2, a, X)
            
            x = RESHAPE(X, [size(T2, 4), size(a, 1), size(a, 1), size(T1, 1)]);

            temp = tensorprod(conj(T2), x, 4, 1, NumDimensionsA=4);
            temp = tensorprod(temp, a, [2, 4], [4, 1], NumDimensionsA=6);
            temp = tensorprod(temp, conj(a), [2, 3, 7], [4, 1, 5], NumDimensionsA=7);
            temp = tensorprod(temp, T1, [2, 3, 5], [1, 2, 3], NumDimensionsA=6);


            temp = RESHAPE(temp, [size(T2, 4) * size(a, 1) * size(a, 1) * size(T1, 1), 1]);
        end

        function temp = FindFixedPointRight1Fun(~, T1, T2, a, X)
            
            x = RESHAPE(X, [size(T1, 4), size(a, 1), size(a, 1), size(T2, 1)]);

            temp = tensorprod(T1, x, 4, 1, NumDimensionsA=4);
            temp = tensorprod(temp, a, [2, 4], [2, 3], NumDimensionsA=6);
            temp = tensorprod(temp, conj(a), [2, 3, 7], [2, 3, 5], NumDimensionsA=7);
            temp = tensorprod(temp, conj(T2), [2, 4, 6], [1, 2, 3], NumDimensionsA=6);


            temp = RESHAPE(temp, [size(T1, 4) * size(a, 3) * size(a, 3) * size(T2, 1), 1]);
        end

        function [A, lambda] = FindFixedPoint(obj, T1, T2, T3, T4, a, b, which, v0, LanczosD)
            
            % Find left or right fixed point of the struction

            %  ---T1---T2---
            %     |    |
            %  ---A----B----
            %     |    |
            %  ---T3*--T4*--

            if which == "left"
                fun = @(x)obj.FindFixedPointLeftFun(T1, T2, T3, T4, a, b, x);
                if v0 == 0
                    [A, lambda] = eigs(fun, size(T3, 4) * size(a, 1) * size(a, 1) * size(T1, 1), 1, 'largestabs', 'Tolerance',1e-10,'MaxIterations',100, ...
                    'SubspaceDimension', LanczosD, 'Display', 0);
                else
                    [A, lambda] = eigs(fun, size(T3, 4) * size(a, 1) * size(a, 1) * size(T1, 1), 1, 'largestabs', 'Tolerance',1e-10,'MaxIterations',100, 'StartVector', v0, 'SubspaceDimension', LanczosD, 'Display', 0);
                end
                
                % fprintf("Leading eigenvalue: %.9e\n", lambda);
                % A = A / norm(A(:));
                A = RESHAPE(A, [size(T3, 4), size(a, 1), size(a, 1), size(T1, 1)]);
                A_temp = RESHAPE(permute(A, [1, 2, 4, 3]), [size(T3, 4) * size(a, 1), size(a, 1) * size(T1, 1)]);
                A = A / trace(A_temp);
            elseif which == "right"
                fun = @(x)obj.FindFixedPointRightFun(T1, T2, T3, T4, a, b, x);
                if v0 == 0
                    [A, lambda] = eigs(fun, size(T2, 4) * size(b, 1) * size(b, 1) * size(T4, 1), 1, 'largestabs', 'Tolerance',1e-10,'MaxIterations',100, ...
                    'SubspaceDimension', LanczosD, 'Display', 0);
                else
                    [A, lambda] = eigs(fun, size(T2, 4) * size(b, 1) * size(b, 1) * size(T4, 1), 1, 'largestabs', 'Tolerance',1e-10,'MaxIterations',100, 'StartVector', v0, ...
                    'SubspaceDimension', LanczosD, 'Display', 0);
                end
                % fprintf("Leading eigenvalue: %.9e\n", lambda);
                % A = A / norm(A(:));
                A = RESHAPE(A, [size(T2, 4), size(b, 1), size(b, 1), size(T4, 1)]);
                A_temp = RESHAPE(permute(A, [1, 2, 4, 3]), [size(T2, 4) * size(b, 1), size(b, 1) * size(T4, 1)]);
                A = A / trace(A_temp);
            end
        end

        function result = BoundaryOverlap(obj, T1, T2, T3, T4, LanczosD)
            
            temp_a = eye(size(T1, 2), size(T3, 2));
            temp_b = eye(size(T2, 2), size(T4, 2));
            temp_a = RESHAPE(temp_a, [1, size(T1, 2), 1, size(T3, 2), 1]);
            temp_b = RESHAPE(temp_b, [1, size(T2, 2), 1, size(T4, 2), 1]);

            
            lfp = FindFixedPoint(obj, T1, T2, T3, T4, temp_a, temp_b, "left", 0, LanczosD);
            rfp = FindFixedPoint(obj, T1, T2, T3, T4, temp_a, temp_b, "right", 0, LanczosD);

            lfp = RESHAPE(lfp, [size(T3, 4), size(T1, 1)]);
            rfp = RESHAPE(rfp, [size(T2, 4), size(T4, 1)]);

            result = tensorprod(lfp, rfp, [1, 2], [2, 1]);
        end

        function A = FindFixedPoint1(obj, T1, T2, a, which)
            
            % Find left or right fixed point of the struction

            %  ---T1---
            %     |    
            %  ---A----
            %     |    
            %  ---T2*--

            if which == "left"
                fun = @(x)obj.FindFixedPointLeft1Fun(T1, T2, a, x);
                [A, ~] = eigs(fun, size(T2, 4) * size(a, 1) * size(a, 1) * size(T1, 1), 1, 'largestabs', 'Tolerance',1e-20,'MaxIterations',1000);
                A = A / norm(A(:));
                A = RESHAPE(A, [size(T2, 4), size(a, 1), size(a, 1), size(T1, 1)]);
            elseif which == "right"
                fun = @(x)obj.FindFixedPointRight1Fun(T1, T2, a, x);
                [A, ~] = eigs(fun, size(T1, 4) * size(a, 1) * size(a, 1) * size(T2, 1), 1, 'largestabs', 'Tolerance',1e-20,'MaxIterations',1000);
                A = A / norm(A(:));
                A = RESHAPE(A, [size(T1, 4), size(a, 1), size(a, 1), size(T2, 1)]);
            end
        end

        function error = CTMError(~, a, b)
            
            if size(a) ~= size(b)
                error = inf;
                return
            end
            error = -1;

            for ii = 1:4
                sa = svd(a{ii}, "econ", "vector");
                sb = svd(b{ii}, "econ", "vector");
                sa = transpose(sa / sa(1));
                sb = transpose(sb / sb(1));
                if numel(sa) > numel(sb)
                    sb = [sb, zeros(1, numel(sa) - numel(sb))];
                elseif numel(sb) > numel(sa)
                    sa = [sa, zeros(1, numel(sb) - numel(sa))];
                end
                temp_error = norm(sa - sb);
                if temp_error > error
                    error = temp_error;
                end
            end
        end

        function [AL, AR, BL, BR] = InitializeVUMPSId(~, chi, D)
            AL = eye(chi * D * D, chi);
            BL = eye(chi * D * D, chi);
            AR = eye(chi * D * D, chi);
            BR = eye(chi * D * D, chi);


            AL = RESHAPE(AL, [chi, D, D, chi]);
            AR = RESHAPE(AR, [chi, D, D, chi]);
            AR = permute(AR, [4, 2, 3, 1]);
            BL = RESHAPE(BL, [chi, D, D, chi]);
            BR = RESHAPE(BR, [chi, D, D, chi]);
            BR = permute(BR, [4, 2, 3, 1]);
        end

        function [AL, AR, BL, BR] = InitializeVUMPS(~, chi, D, a, b)

            AL = zeros(chi, D, D, chi);
            BL = zeros(chi, D, D, chi);

            a_temp = a(1:1, 1:size(a, 2), 1:1, 1:size(a, 4), 1:size(a, 5));
            b_temp = b(1:1, 1:size(b, 2), 1:1, 1:size(b, 4), 1:size(b, 5));

            AL_temp = tensorprod(b_temp, conj(b_temp), [2, 5], [2, 5], NumDimensionsA=5);
            % % disp(size(AL_temp))
            AL_temp = permute(AL_temp, [1, 4, 3, 6, 2, 5]);
            AL_temp = RESHAPE(AL_temp, [size(AL_temp, 1) * size(AL_temp, 2), size(AL_temp, 3), size(AL_temp, 4), size(AL_temp, 5) * size(AL_temp, 6)]);
            AL(1:size(AL_temp, 1), 1:size(AL_temp, 2), 1:size(AL_temp, 3), 1:size(AL_temp, 4)) = AL_temp;
            % % disp(size(AL))
            AR = AL;

            BL_temp = tensorprod(a_temp, conj(a_temp), [2, 5], [2, 5], NumDimensionsA=5);
            % % disp(size(BL_temp))
            BL_temp = permute(BL_temp, [1, 4, 3, 6, 2, 5]);
            BL_temp = RESHAPE(BL_temp, [size(BL_temp, 1) * size(BL_temp, 2), size(BL_temp, 3), size(BL_temp, 4), size(BL_temp, 5) * size(BL_temp, 6)]);
            BL(1:size(BL_temp, 1), 1:size(BL_temp, 2), 1:size(BL_temp, 3), 1:size(BL_temp, 4)) = BL_temp;
            % % disp(size(BL))
            BR = BL;

        end

        function [AL, AR, BL, BR] = InitializeVUMPSCTMRG(~, chi, D, TA, TB)
            AL = zeros(chi, D, D, chi);
            BL = zeros(chi, D, D, chi);
            AR = zeros(chi, D, D, chi);
            BR = zeros(chi, D, D, chi);

            AL(1:size(TA, 1), 1:size(AL, 2), 1:size(AL, 3), 1:size(TA, 4)) = TA;
            AR(1:size(TA, 1), 1:size(AL, 2), 1:size(AL, 3), 1:size(TA, 4)) = TA;
            
            BL(1:size(TB, 1), 1:size(BL, 2), 1:size(BL, 3), 1:size(TB, 4)) = TB;
            BR(1:size(TB, 1), 1:size(BL, 2), 1:size(BL, 3), 1:size(TB, 4)) = TB;
        end

        function [U, P] = PolarDecomp(~, a, method, which)
            if method == "svd"
                [u, s, v] = svd(a, "econ");
                if which == "left"
                    U = u * (v');
                    P = v * s * (v');
                elseif which == "right"
                    P = u * s * (u');
                    U = u * (v');
                end
            end
        end

        function newBc = BuildBC(~, L, TA, a, R)
            newBc = tensorprod(L, TA, 4, 1, NumDimensionsA=4);
            newBc = tensorprod(newBc, a, [2, 4], [1, 2], NumDimensionsA=6);
             newBc = tensorprod(newBc, conj(a), [2, 3, 7], [1, 2, 5], NumDimensionsA=7);
            newBc = tensorprod(newBc, R, [2, 3, 5], [1, 2, 3], NumDimensionsA=6);
        end

        function newAc = BuildAC(~, L, TA, a, R)
            newAc = tensorprod(L, TA, 4, 1, NumDimensionsA=4);
            newAc = tensorprod(newAc, a, [2, 4], [1, 2], NumDimensionsA=6);
            newAc = tensorprod(newAc, conj(a), [2, 3, 7], [1, 2, 5], NumDimensionsA=7);
            newAc = tensorprod(newAc, R, [2, 3, 5], [1, 2, 3], NumDimensionsA=6);
        end

        function newCBA = BuildCBA(~, L, R)
            newCBA = tensorprod(L, R, [2, 3, 4], [2, 3, 1]);
        end

        function v0 = Buildv0(obj, a, whichone)
            if whichone == "left"
                v0 = tensorprod(a, conj(a), [1, 5], [1, 5]);
                v0 = permute(v0, [3, 6, 2, 5, 1, 4]);
                v0 = RESHAPE(v0, [size(a, 4) * size(a, 4), size(a, 3) * size(a, 3), size(a, 2) * size(a, 2)]);
            elseif whichone == "right"
                v0 = tensorprod(a, conj(a), [3, 5], [3, 5]);
                v0 = permute(v0, [2, 5, 1, 4, 3, 6]);
                v0 = RESHAPE(v0, [size(a, 2) * size(a, 2), size(a, 1) * size(a, 1), size(a, 4) * size(a, 4)]);
            end
            if size(a,4) * size(a, 4) >= obj.chimax
                v0 = v0(1:obj.chimax, 1:size(v0, 2), 1:obj.chimax);
            else
                v0_temp = zeros(obj.chimax, size(v0, 2), obj.chimax);
                v0_temp(1:size(v0, 1), 1:size(v0, 2), 1:size(v0, 3)) = v0;
                v0 = v0_temp;
            end
            v0 = v0(:);
        end

        function [MPSA_left, MPSB_left] = BuildLeftCanonicalForm(~, MPSA, MPSB, tol, maxstep)

            MPSA_left = MPSA;
            MPSB_left = MPSB;

            RA_error = inf;
            RB_error = inf;
            RA_old = eye(size(MPSA, 3), size(MPSB, 1));
            RB_old = eye(size(MPSB, 3), size(MPSA, 1));

            step = 0;

            while ((RA_error > tol) || (RB_error > tol))

                if step > maxstep
                    break;
                end

                [MPSA_left, RA_new] = qr(RESHAPE(tensorprod(RB_old, MPSA_left, 2, 1), ...
                                        [size(MPSA_left, 1) * size(MPSA_left, 2), size(MPSA_left, 3)]), "econ");
                MPSA_left = RESHAPE(MPSA_left, [size(MPSA, 1), size(MPSA, 2), size(MPSA, 3)]);
                [MPSB_left, RB_new] = qr(RESHAPE(tensorprod(RA_new, MPSB_left, 2, 1), ...
                                        [size(MPSB_left, 1) * size(MPSB_left, 2), size(MPSB_left, 3)]), "econ");
                MPSB_left = RESHAPE(MPSB_left, [size(MPSB, 1), size(MPSB, 2), size(MPSB, 3)]);

                RA_error = norm(RA_old - RA_new);
                RB_error = norm(RB_old - RB_new);

                RA_old = RA_new;
                RB_old = RB_new;

                step = step + 1;

                % fprintf("%.9e, %.9e\n", RA_error, RB_error);
            end

        end

        function [MPSA_right, MPSB_right] = BuildRightCanonicalForm(~, MPSA, MPSB, tol, maxstep)

            MPSA_right = permute(MPSA, [3, 2, 1]);
            MPSB_right = permute(MPSB, [3, 2, 1]);

            RA_error = inf;
            RB_error = inf;
            RA_old = eye(size(MPSA, 1), size(MPSB, 3));
            RB_old = eye(size(MPSB, 1), size(MPSA, 3));

            step = 0;

            while ((RA_error > tol) || (RB_error > tol))

                if step > maxstep
                    break;
                end

                [MPSA_right, RA_new] = qr(RESHAPE(tensorprod(RB_old, MPSA_right, 2, 1), ...
                                        [size(MPSA_right, 1) * size(MPSA_right, 2), size(MPSA_right, 3)]), "econ");
                MPSA_right = RESHAPE(MPSA_right, [size(MPSA, 3), size(MPSA, 2), size(MPSA, 1)]);
                [MPSB_right, RB_new] = qr(RESHAPE(tensorprod(RA_new, MPSB_right, 2, 1), ...
                                        [size(MPSB_right, 1) * size(MPSB_right, 2), size(MPSB_right, 3)]), "econ");
                MPSB_right = RESHAPE(MPSB_right, [size(MPSB, 3), size(MPSB, 2), size(MPSB, 1)]);

                RA_error = norm(RA_old - RA_new);
                RB_error = norm(RB_old - RB_new);

                RA_old = RA_new;
                RB_old = RB_new;

                step = step + 1;

                % fprintf("%.9e, %.9e\n", RA_error, RB_error);
            end

            MPSA_right = permute(MPSA_right, [3, 2, 1]);
            MPSB_right = permute(MPSB_right, [3, 2, 1]);

        end



        function [newAL, newAR, newBL, newBR] =  VUMPS(obj, AL, AR, BL, BR, a, b, whichone, vumps_err, LanczosD)

            oldAL = AL;
            oldBL = BL;
            oldAR = AR;
            oldBR = BR;

            newAL = AL;
            newBL = BL;
            newAR = AR;
            newBR = BR;

            % AL_tA = tensorprod(tensorprod(AL, a, 2, 2, NumDimensionsA=4), conj(a), [2, 7], [2, 5], NumDimensionsA=7);
            % BL_tB = tensorprod(tensorprod(BL, b, 2, 2, NumDimensionsA=4), conj(b), [2, 7], [2, 5], NumDimensionsA=7);

            % AL_tA_L = permute(AL_tA, [1, 3, 6, 5, 8, 2, 4, 7]);
            % AL_tA_L = RESHAPE(AL_tA_L, [size(AL, 1) * size(a, 1) * size(a, 1), size(a, 4) * size(a, 4), size(AL, 4) * size(a, 3) * size(a, 3)]);

            % BL_tB_L = permute(BL_tB, [1, 3, 6, 5, 8, 2, 4, 7]);
            % BL_tB_L = RESHAPE(BL_tB_L, [size(BL, 1) * size(b, 1) * size(b, 1), size(b, 4) * size(b, 4), size(BL, 4) * size(b, 3) * size(b, 3)]);

            % RA_error = inf;
            % RB_error = inf;
            % RB_left_old = eye(size(BL, 4) * size(b, 3) * size(b, 3), size(AL, 1) * size(a, 1) * size(a, 1));
            % RA_left_old = eye(size(AL, 4) * size(a, 3) * size(a, 3), size(BL, 1) * size(b, 1) * size(b, 1));

            % step = 1;
            
            % while ((RA_error > 1e-5) || (RB_error > 1e-5))

            %     if step > 20
            %         break;
            %     end
            %     % % disp(size(RB_left_old))
            %     % % disp(size(AL_tA_L))
            %     [AL_tA_L, RA_left_new] = qr(RESHAPE(tensorprod(RB_left_old, AL_tA_L, 2, 1), ...
            %                             [size(AL_tA_L, 1) * size(AL_tA_L, 2), size(AL_tA_L, 3)]), "econ");
            %     AL_tA_L = RESHAPE(AL_tA_L, [size(AL, 1) * size(a, 1) * size(a, 1), size(a, 4) * size(a, 4), size(AL, 4) * size(a, 3) * size(a, 3)]);
            %     [BL_tB_L, RB_left_new] = qr(RESHAPE(tensorprod(RA_left_new, BL_tB_L, 2, 1), ...
            %                             [size(BL_tB_L, 1) * size(BL_tB_L, 2), size(BL_tB_L, 3)]), "econ");
            %     BL_tB_L = RESHAPE(BL_tB_L, [size(BL, 1) * size(b, 1) * size(b, 1), size(b, 4) * size(b, 4), size(BL, 4) * size(b, 3) * size(b, 3)]);

            %     RA_error = norm(RA_left_old - RA_left_new);
            %     RB_error = norm(RB_left_old - RB_left_new);

            %     RA_left_old = RA_left_new;
            %     RB_left_old = RB_left_new;

            %     step = step + 1;

            %     % fprintf("%.9e, %.9e\n", RA_error, RB_error);
            % end

            % AL_tA_R = permute(AL_tA, [2, 4, 7, 5, 8, 1, 3, 6]);
            % AL_tA_R = RESHAPE(AL_tA_R, [size(AL, 4) * size(a, 3) * size(a, 3), size(a, 4) * size(a, 4), size(AL, 1) * size(a, 1) * size(a, 1)]);           

            % BL_tB_R = permute(BL_tB, [2, 4, 7, 5, 8, 1, 3, 6]);
            % BL_tB_R = RESHAPE(BL_tB_R, [size(BL, 4) * size(b, 3) * size(b, 3), size(b, 4) * size(b, 4), size(BL, 1) * size(b, 1) * size(b, 1)]);

            % RA_error = inf;
            % RB_error = inf;
            % RB_right_old = eye(size(BL, 1) * size(b, 1) * size(b, 1), size(AL, 4) * size(a, 3) * size(a, 3));
            % RA_right_old = eye(size(AL, 1) * size(a, 1) * size(a, 1), size(BL, 4) * size(b, 3) * size(b, 3));

            % step = 1;

            % while ((RA_error > 1e-5) || (RB_error > 1e-5))
            %     if step > 20
            %         break;
            %     end
            %     % % disp(size(RB_right_old))
            %     % % disp(size(AL_tA_R))
            %     [AL_tA_R, RA_right_new] = qr(RESHAPE(tensorprod(RB_right_old, AL_tA_R, 2, 1), ...
            %                             [size(AL_tA_R, 1) * size(AL_tA_R, 2), size(AL_tA_R, 3)]), "econ");
            %     AL_tA_R = RESHAPE(AL_tA_R, [size(AL, 4) * size(a, 3) * size(a, 3), size(a, 4) * size(a, 4), size(AL, 1) * size(a, 1) * size(a, 1)]);
            %     [BL_tB_R, RB_right_new] = qr(RESHAPE(tensorprod(RA_right_new, BL_tB_R, 2, 1), ...
            %                             [size(BL_tB_R, 1) * size(BL_tB_R, 2), size(BL_tB_R, 3)]), "econ");
            %     BL_tB_R = RESHAPE(BL_tB_R, [size(BL, 4) * size(b, 3) * size(b, 3), size(b, 4) * size(b, 4), size(BL, 1) * size(b, 1) * size(b, 1)]);

            %     RA_error = norm(RA_right_old - RA_right_new);
            %     RB_error = norm(RB_right_old - RB_right_new);

            %     RA_right_old = RA_right_new;
            %     RB_right_old = RB_right_new;


            %     % fprintf("%.9e, %.9e\n", RA_error, RB_error);
            %     step = step + 1;
            % end

            % CAB_new = RA_left_new * transpose(RB_right_new);
            % CBA_new = RB_left_new * transpose(RA_right_new);

            % % fprintf("%.6e \n", eig(CAB_new))
            % % fprintf("----------------------------------------------\n")
            % % fprintf("%.6e \n", eig(CBA_new))

            % % [uCAB, ~, ~] = svds(CAB_new, obj.chimax);
            % % [uCBA, ~, ~] = svds(CBA_new, obj.chimax);

            % [uCAB, ~, ~] = svds(CAB_new, obj.chimax);
            % % uCAB = uCAB(:, 1:obj.chimax);
            % [uCBA, ~, ~] = svds(CBA_new, obj.chimax);
            % % uCBA = uCBA(:, 1:obj.chimax);


            % oldBL = tensorprod(tensorprod(uCBA', AL_tA_L, 2, 1), uCAB, 3, 1);
            % oldAL = tensorprod(tensorprod(uCAB', BL_tB_L, 2, 1), uCBA, 3, 1);

            % RA_old = eye(size(AL, 4), size(BL, 1));
            % RB_old = eye(size(BL, 4), size(AL, 1));
            % RA_error = inf;
            % RB_error = inf;

            % step = 1;
            % while ((RA_error > 1e-5) || (RB_error > 1e-5))
            %     if step > 20
            %         break;
            %     end
            %     [oldAL, RA_new] = qr(RESHAPE(tensorprod(RB_old, oldAL, 2, 1), ...
            %                             [size(oldAL, 1) * size(oldAL, 2), size(oldAL, 3)]), "econ");
            %     oldAL = RESHAPE(oldAL, [size(AL, 4), size(a, 4) * size(a, 4), size(AL, 1)]);
            %     [oldBL, RB_new] = qr(RESHAPE(tensorprod(RA_new, oldBL, 2, 1), ...
            %                          [size(oldBL, 1) * size(oldBL, 2), size(oldBL, 3)]), "econ");
            %     oldBL = RESHAPE(oldBL, [size(BL, 4), size(b, 4) * size(b, 4), size(BL, 1)]);

            %     RA_error = norm(RA_old - RA_new);
            %     RB_error = norm(RB_old - RB_new);

            %     RA_old = RA_new;
            %     RB_old = RB_new;

            %     step = step + 1;
            %     % fprintf("%.9e, %.9e\n", RA_error, RB_error);
            % end

            % oldBL = RESHAPE(oldBL, [size(BL, 1), size(BL, 2), size(BL, 3), size(BL, 4)]);
            % oldAL = RESHAPE(oldAL, [size(AL, 1), size(AL, 2), size(AL, 3), size(AL, 4)]);

            % newAL = oldAL;
            % newBL = oldBL;
            % newAR = oldAR;
            % newBR = oldBR;

            i = 1;

            while true

                % fprintf("Iteration #%d\n", i);

                % fprintf("Finding left fixed point of AB\n")

                % LanczosD = 10;

                % v0 = obj.Buildv0(b, "left");
                v0 = 0;
                [AB_lfp, eigl] = obj.FindFixedPoint(AL, BL, permute(oldBL, [4, 2, 3, 1]), permute(oldAL, [4, 2, 3, 1]), a, b, "left", v0, LanczosD);
                

                % fprintf("Finding right fixed point of AB\n")

                % v0 = obj.Buildv0(a, "right");
                v0 = 0;
                [AB_rfp, eigr] = obj.FindFixedPoint(AL, BL, permute(oldBR, [4, 2, 3, 1]), permute(oldAR, [4, 2, 3, 1]), a, b, "right", v0, LanczosD);





                % while (abs((eigl - eigr) / eigl) > 1e-6) && (LanczosD <= 100)
                %     LanczosD = LanczosD + 50;
                %     % v0 = AB_lfp(:);
                %     [AB_lfp, eigl] = obj.FindFixedPoint(AL, BL, permute(oldBL, [4, 2, 3, 1]), permute(oldAL, [4, 2, 3, 1]), a, b, "left", v0, LanczosD);
                %     % v0 = AB_rfp(:);
                %     [AB_rfp, eigr] = obj.FindFixedPoint(AL, BL, permute(oldBR, [4, 2, 3, 1]), permute(oldAR, [4, 2, 3, 1]), a, b, "right", v0, LanczosD);
                % end

                % fprintf("Left and right eigenvalue relative error: %.9e\n", abs((eigl - eigr) / eigl));
                % fprintf("Finding left fixed point of BA\n")
                % BA_lfp = obj.FindFixedPoint(BL, AL, permute(oldAL, [4, 2, 3, 1]), permute(oldBL, [4, 2, 3, 1]), b, a, "left");
                BA_lfp = obj.FindFixedPointLeft1Fun(AL, permute(oldBL, [4, 2, 3, 1]), a, AB_lfp);
                BA_lfp = RESHAPE(BA_lfp, [size(oldBL, 1), size(a, 1), size(a, 1), size(AL, 1)]);
                % BA_lfp = BA_lfp / norm(BA_lfp(:));
                % fprintf("Finding right fixed point of BA\n")
                % AB_rfp = obj.FindFixedPoint(AL, BL, permute(oldBR, [4, 2, 3, 1]), permute(oldAR, [4, 2, 3, 1]), a, b, "right");
                BA_rfp = obj.FindFixedPointRight1Fun(BL, permute(oldAR, [4, 2, 3, 1]), b, AB_rfp);
                BA_rfp = RESHAPE(BA_rfp, [size(BL, 4), size(b, 3), size(b, 3), size(oldAR, 4)]);
                % BA_rfp = BA_rfp / norm(AB_rfp(:));

                % fprintf("Update AC and CAB\n")
                newAC = obj.BuildBC(BA_lfp, BL, b, AB_rfp);
                norm_newAC = sqrt(tensorprod(newAC, conj(newAC), [1,2,3,4], [1,2,3,4], NumDimensionsA=4));
                newAC = newAC / norm_newAC;
                newCAB = obj.BuildCBA(AB_lfp, AB_rfp);
                newCAB = newCAB / norm(newCAB(:));


                % fprintf("Update BC and CBA\n")
                newBC = obj.BuildBC(AB_lfp, AL, a, BA_rfp);
                norm_newBC = sqrt(tensorprod(newBC, conj(newBC), [1,2,3,4], [1,2,3,4], NumDimensionsA=4));
                newBC = newBC / norm_newBC;
                newCBA = obj.BuildCBA(BA_lfp / norm(BA_lfp(:)), BA_rfp / norm(BA_rfp(:)));
                newCBA = newCBA / norm(newCBA(:));

                [UBC_left, ~] = obj.PolarDecomp(RESHAPE(newBC, [size(newBC, 1) * size(newBC, 2) * size(newBC, 3), size(newBC, 4)]), "svd", "left");
                [UBC_right, ~] = obj.PolarDecomp(transpose(RESHAPE(newBC, [size(newBC, 1), size(newBC, 2) * size(newBC, 3) * size(newBC, 4)])), "svd", "left");
                UBC_right = transpose(UBC_right);
                
                [UAC_left, ~] = obj.PolarDecomp(RESHAPE(newAC, [size(newAC, 1) * size(newAC, 2) * size(newAC, 3), size(newAC, 4)]), "svd", "left");
                [UAC_right, ~] = obj.PolarDecomp(transpose(RESHAPE(newAC, [size(newAC, 1), size(newAC, 2) * size(newAC, 3) * size(newAC, 4)])), "svd", "left");
                UAC_right = transpose(UAC_right);

                [UCAB_left, ~] = obj.PolarDecomp(newCAB, "svd", "left");
                [UCBA_left, ~] = obj.PolarDecomp(newCBA, "svd", "left");

                [UCAB_right, ~] = obj.PolarDecomp(transpose(newCAB), "svd", "left");
                UCAB_right = transpose(UCAB_right);
                [UCBA_right, ~] = obj.PolarDecomp(transpose(newCBA), "svd", "left");
                UCBA_right = transpose(UCBA_right);

                if whichone == "a"
                    % fprintf("Update AL, AR\n")
                    newAL = UAC_left * (UCAB_left');
                    newAR = (UCBA_right') * UAC_right;
                    err_l = norm(newAL * newCAB - RESHAPE(newAC, [size(newAC, 1) * size(newAC, 2) * size(newAC, 3), size(newAC, 4)]), 2);
                    err_r = norm(newCBA * newAR - RESHAPE(newAC, [size(newAC, 1), size(newAC, 2) * size(newAC, 3) * size(newAC, 4)]), 2);
                    newAL = RESHAPE(newAL, [size(newAC, 1), size(newAC, 2), size(newAC, 3), size(newAC, 4)]);
                    newAR = RESHAPE(newAR, [size(newAC, 1), size(newAC, 2), size(newAC, 3), size(newAC, 4)]);
                    err = max([err_l, err_r]);

                    % fprintf("[errL, errR]: %.9e, %.9e\n", err_l, err_r);
                elseif whichone == "b"
                    % fprintf("Update BL, BR\n")
                    newBL = UBC_left * (UCBA_left');
                    newBR = (UCAB_right') * UBC_right;
                    err_l = norm(newBL * newCBA - RESHAPE(newBC, [size(newBC, 1) * size(newBC, 2) * size(newBC, 3), size(newBC, 4)]), 2);
                    err_r = norm(newCAB * newBR - RESHAPE(newBC, [size(newBC, 1), size(newBC, 2) * size(newBC, 3) * size(newBC, 4)]), 2);
                    newBL = RESHAPE(newBL, [size(newBC, 1), size(newBC, 2), size(newBC, 3), size(newBC, 4)]);
                    newBR = RESHAPE(newBR, [size(newBC, 1), size(newBC, 2), size(newBC, 3), size(newBC, 4)]);
                    err = max([err_l, err_r]);

                    % fprintf("[errL, errR]: %.9e, %.9e\n", err_l, err_r);
                elseif whichone == "both"
                    % fprintf("Update AL, AR\n")
                    newAL = UAC_left * (UCAB_left');
                    newAR = (UCBA_right') * UAC_right;
                    err_Al = norm(newAL * newCAB - RESHAPE(newAC, [size(newAC, 1) * size(newAC, 2) * size(newAC, 3), size(newAC, 4)]), 2);
                    err_Ar = norm(newCBA * newAR - RESHAPE(newAC, [size(newAC, 1), size(newAC, 2) * size(newAC, 3) * size(newAC, 4)]), 2);
                    newAL = RESHAPE(newAL, [size(newAC, 1), size(newAC, 2), size(newAC, 3), size(newAC, 4)]);
                    newAR = RESHAPE(newAR, [size(newAC, 1), size(newAC, 2), size(newAC, 3), size(newAC, 4)]);
                    % fprintf("Update BL, BR\n")
                    newBL = UBC_left * (UCBA_left');
                    newBR = (UCAB_right') * UBC_right;
                    err_Bl = norm(newBL * newCBA - RESHAPE(newBC, [size(newBC, 1) * size(newBC, 2) * size(newBC, 3), size(newBC, 4)]), 2);
                    err_Br = norm(newCAB * newBR - RESHAPE(newBC, [size(newBC, 1), size(newBC, 2) * size(newBC, 3) * size(newBC, 4)]), 2);
                    newBL = RESHAPE(newBL, [size(newBC, 1), size(newBC, 2), size(newBC, 3), size(newBC, 4)]);
                    newBR = RESHAPE(newBR, [size(newBC, 1), size(newBC, 2), size(newBC, 3), size(newBC, 4)]);
                    err = max([err_Al, err_Ar, err_Bl, err_Br]);
                    % fprintf("[errAL, errAR, errBL, errBR]: %.9e, %.9e, %.9e, %.9e\n", err_Al, err_Ar, err_Bl, err_Br);
                end

                % newAL = tensorprod(newAC, pinv(newCAB), 4, 1);
                % newBL = tensorprod(newBC, pinv(newCBA), 4, 1);
                % newAR = tensorprod(pinv(newCBA), newAC, 2, 1);
                % newBR = tensorprod(pinv(newCAB), newBC, 2, 1);

                
                % errA_l = norm(RESHAPE(newAL, [size(newAC, 1) * size(newAC, 2) * size(newAC, 3), size(newAC, 4)]) * newCAB - ...
                %              RESHAPE(newAC, [size(newAC, 1) * size(newAC, 2) * size(newAC, 3), size(newAC, 4)]), 2);
                % errA_r = norm(newCBA * RESHAPE(newAR, [size(newAC, 1), size(newAC, 2) * size(newAC, 3) * size(newAC, 4)])- ...
                %               RESHAPE(newAC, [size(newAC, 1), size(newAC, 2) * size(newAC, 3) * size(newAC, 4)]), 2);
                % errB_l = norm(RESHAPE(newBL, [size(newBC, 1) * size(newBC, 2) * size(newBC, 3), size(newBC, 4)]) * newCBA - ...
                %              RESHAPE(newBC, [size(newBC, 1) * size(newBC, 2) * size(newBC, 3), size(newBC, 4)]), 2);
                % errB_r = norm(newCAB * RESHAPE(newBR, [size(newBC, 1), size(newBC, 2) * size(newBC, 3) * size(newBC, 4)]) - ...
                %               RESHAPE(newBC, [size(newBC, 1), size(newBC, 2) * size(newBC, 3) * size(newBC, 4)]), 2);

                if err < vumps_err
                    break
                else
                    if whichone == "a"
                        oldAL = newAL;
                        oldAR = newAR;
                    elseif whichone == "b"
                        oldBL = newBL;
                        oldBR = newBR;
                    elseif whichone == "both"
                        oldAL = newAL;
                        oldAR = newAR;
                        oldBL = newBL;
                        oldBR = newBR;
                    end
                    i = i + 1;
                end

                if i > 100
                    % fprintf("VUMPS reaches maximum iteration!\n");
                    break
                end

            end
        end

        function rho = CorrelationFunctionDM(obj, a, b, dis, LFP, RFP, AL_u, BL_u, AL_d, BL_d)
            temp = tensorprod(AL_d, LFP, 4, 1, NumDimensionsA=4);
            temp = tensorprod(temp, a, [2, 4], [4, 1], NumDimensionsA=6);
            temp = tensorprod(temp, conj(a), [2, 3], [4, 1], NumDimensionsA=7);
            temp = tensorprod(temp, AL_u, [2, 3, 6], [1, 2, 3], NumDimensionsA=8);
            temp = permute(temp, [3, 5, 1, 2, 4, 6]);

            a_or_b = 1;

            for ii = 1:dis
                if a_or_b == 1
                    L_u = BL_u;
                    lattice = b;
                    L_d = BL_d;
                else
                    L_u = AL_u;
                    lattice = a;
                    L_d = AL_d;
                end

                temp = tensorprod(L_d, temp, 4, 3, NumDimensionsA=4);
                temp = tensorprod(temp, lattice, [2, 6], [4, 1], NumDimensionsA=8);
                if ii ~= dis
                    temp = tensorprod(temp, conj(lattice), [2, 5, 9], [4, 1, 5], NumDimensionsA=9);
                    temp = tensorprod(temp, L_u, [4, 5, 7], [1, 2, 3], NumDimensionsA=8);
                    temp = permute(temp, [2, 3, 1, 4, 5, 6]);
                else
                    temp = tensorprod(temp, conj(lattice), [2, 5], [4, 1], NumDimensionsA=9);
                    temp = tensorprod(temp, L_u, [4, 5, 8], [1, 2, 3], NumDimensionsA=10);
                    temp = permute(temp, [2, 3, 5, 7, 1, 4, 6, 8]);
                end

                a_or_b = mod(a_or_b + 1, 2);
            end

            if a_or_b == 1
                temp = tensorprod(BL_d, temp, 4, 5, NumDimensionsA=4);
                temp = tensorprod(temp, b, [2, 8], [4, 1], NumDimensionsA=10);
                temp = tensorprod(temp, conj(b), [2, 7, 11], [4, 1, 5], NumDimensionsA=11);
                temp = tensorprod(temp, BL_u, [6, 7, 9], [1, 2, 3], NumDimensionsA=10);
                temp = permute(temp, [2, 3, 4, 5, 1, 6, 7, 8]);
            end

            temp = tensorprod(temp, RFP, [5, 6, 7, 8], [4, 2, 3, 1], NumDimensionsA=8);
            temp = RESHAPE(permute(temp, [1, 3, 2, 4]), [obj.d * obj.d, obj.d * obj.d]);
            rho = temp / trace(temp);

            % % disp(rho)
            

        end

        function [dm, Lfix, Rfix] = DensityMatrixVUMPS(obj, AL_u, BL_u, AL_d, BL_d, a, b, LanczosD)

            % v0 = obj.Buildv0(obj.b, "left");
            v0 = 0;
            [Lfix, eigl] = obj.FindFixedPoint(AL_u, BL_u, conj(AL_d), conj(BL_d), a, b, "left", v0, LanczosD);
            % v0 = obj.Buildv0(obj.a, "right");
            v0 = 0;
            [Rfix, eigr] = obj.FindFixedPoint(AL_u, BL_u, conj(AL_d), conj(BL_d), a, b, "right", v0, LanczosD);

            % while (abs((eigl - eigr) / eigl) > 1e-6) && (LanczosD <= 100)
            %     LanczosD = LanczosD + 50;
            %     % v0 = Lfix(:);
            %     [Lfix, eigl] = obj.FindFixedPoint(AL_u, BL_u, conj(AL_d), conj(BL_d), a, b, "left", v0, LanczosD);
            %     % v0 = Rfix(:);
            %     [Rfix, eigr] = obj.FindFixedPoint(AL_u, BL_u, conj(AL_d), conj(BL_d), a, b, "right", v0, LanczosD);
            % end
            % fprintf("Left and right eigenvalue relative error: %.9e\n", abs((eigl - eigr) / eigl));
            
            tempL = tensorprod(Lfix, AL_u, 4, 1, NumDimensionsA=4);
            tempL = tensorprod(tempL, a, [2, 4], [1, 2], NumDimensionsA=6);
            tempL = tensorprod(tempL, conj(a), [2, 3], [1, 2], NumDimensionsA=7);

            tempR = tensorprod(BL_u, Rfix, 4, 1, NumDimensionsA=4);
            tempR = tensorprod(tempR, b, [2, 4], [2, 3], NumDimensionsA=6);
            tempR = tensorprod(tempR, conj(b), [2, 3], [2, 3], NumDimensionsA=7);

            temp = tensorprod(tempL, tempR, [2, 3, 6], [1, 3, 6], NumDimensionsA=8);
            temp = tensorprod(AL_d, temp, [4, 2, 3], [1, 2, 4], NumDimensionsA=4);
            temp = tensorprod(BL_d, temp, [4, 2, 3, 1], [1, 5, 7, 4], NumDimensionsA=4);

            dm = RESHAPE(permute(temp, [1, 3, 2, 4]), [obj.d * obj.d, obj.d * obj.d]);
            dm = dm / trace(dm);
        end

        function [dm, TA1, TB1, TA3, TB3] = VUMPStest(obj, vumps_tol, vumps_e_tol, H, LanczosD)

            % [oldAL_u, oldAR_u, oldBL_u, oldBR_u] = obj.InitializeVUMPSCTMRG(obj.chimax, obj.D, obj.TA{1}, obj.TB{1});
            % [oldAL_d, oldAR_d, oldBL_d, oldBR_d] = obj.InitializeVUMPSCTMRG(obj.chimax, obj.D, obj.TA{3}, obj.TB{3});
            % [oldAL_u, oldAR_u, oldBL_u, oldBR_u] = obj.InitializeVUMPS(obj.chimax, size(obj.a, 1), obj.a, obj.b);
            % [oldAL_d, oldAR_d, oldBL_d, oldBR_d] = obj.InitializeVUMPS(obj.chimax, size(obj.a, 1), permute(obj.a, [3, 4, 1, 2, 5]), permute(obj.b, [3, 4, 1, 2, 5]));

            obj.CTMInitialize(obj, "eye")
            [AL, AR, BL, BR] = obj.InitializeVUMPSId(obj.chimax, size(obj.a, 1));
            
            
            oldAL_d = AL;
            oldBL_d = BL;
            oldAR_d = AR;
            oldBR_d = BR;

            oldAL_u = AL;
            oldBL_u = BL;
            oldAR_u = AR;
            oldBR_u = BR;

            oldAL_r = AL;
            oldBL_r = BL;
            oldAR_r = AR;
            oldBR_r = BR;

            oldAL_l = AL;
            oldBL_l = BL;
            oldAR_l = AR;
            oldBR_l = BR;
            
            
            olddm_ud = ones(4, 4) / 4;
            olddm_lr = ones(4, 4) / 4;
            energy_old_ud = inf;
            energy_old_lr = inf;

            layer = 1;

            while true

                % fprintf("Update upper boundary\n")
                [newAL_u, ~, newBL_u, ~] = obj.VUMPS(oldAL_u, oldAR_u, oldBL_u, oldBR_u, obj.a, obj.b, "both", vumps_tol, LanczosD);
                
                overlap_on_u = obj.BoundaryOverlap(oldAL_u, oldBL_u, permute(newAL_u, [4, 2, 3, 1]), permute(newBL_u, [4, 2, 3, 1]), LanczosD);
                overlap_nn_u = obj.BoundaryOverlap(newAL_u, newBL_u, permute(newAL_u, [4, 2, 3, 1]), permute(newBL_u, [4, 2, 3, 1]), LanczosD);

                oldAL_u = newAL_u;
                oldBL_u = newBL_u;

                TA1 = oldAL_u;
                TB1 = oldBL_u;

                % % fprintf("Update left boundary\n")

                % [newAL_l, newAR_l, newBL_l, newBR_l] = obj.VUMPS(oldAL_l, oldAR_l, oldBL_l, oldBR_l, permute(obj.a, [4, 1, 2, 3, 5]), permute(obj.b, [4, 1, 2, 3, 5]), "both", vumps_tol);
                % oldAL_l = newAL_l;
                % oldBL_l = newBL_l;
                % oldAR_l = newAR_l;
                % oldBR_l = newBR_l;

                % fprintf("Update lower boundary\n")
                [newAL_d, ~, newBL_d, ~] = obj.VUMPS(oldAL_d, oldAR_d, oldBL_d, oldBR_d, permute(obj.a, [3, 4, 1, 2, 5]), permute(obj.b, [3, 4, 1, 2, 5]), "both", vumps_tol, LanczosD);

                overlap_on_d = obj.BoundaryOverlap(permute(newAL_d, [4, 2, 3, 1]), permute(newBL_d, [4, 2, 3, 1]), oldAL_d, oldBL_d, LanczosD);
                overlap_nn_d = obj.BoundaryOverlap(permute(newAL_d, [4, 2, 3, 1]), permute(newBL_d, [4, 2, 3, 1]), newAL_d, newBL_d, LanczosD);

                oldAL_d = newAL_d;
                oldBL_d = newBL_d;

                TA3 = oldAL_d;
                TB3 = oldBL_d;

                % % fprintf("Update right boundary\n")
                % [newAL_r, newAR_r, newBL_r, newBR_r] = obj.VUMPS(oldAL_r, oldAR_r, oldBL_r, oldBR_r, permute(obj.a, [2, 3, 4, 1, 5]), permute(obj.b, [2, 3, 4, 1, 5]), "both", vumps_tol);
                % oldAL_r = newAL_r;
                % oldBL_r = newBL_r;
                % oldAR_r = newAR_r;
                % oldBR_r = newBR_r;


                % [newAL_d, newAR_d, newBL_d, newBR_d] = obj.VUMPS(oldAL_d, oldAR_d, oldBL_d, oldBR_d, permute(obj.a, [3, 4, 1, 2, 5]), permute(obj.b, [3, 4, 1, 2, 5]), "b");
                % oldAL_d = newAL_d;
                % oldBL_d = newBL_d;
                % oldAR_d = newAR_d;
                % oldBR_d = newBR_d;
                
                [dm_ud, oldA_l, oldB_r] = obj.DensityMatrixVUMPS(oldAL_u, oldBL_u, oldAL_d, oldBL_d, obj.a, obj.b, LanczosD);
                
                % Cor_1 = obj.CorrelationFunctionDM(obj.a, obj.b, 1, oldA_l, oldB_r, oldAL_u, oldBL_u, oldAL_d, oldBL_d);
                % Cor_2 = obj.CorrelationFunctionDM(obj.a, obj.b, 2, oldA_l, oldB_r, oldAL_u, oldBL_u, oldAL_d, oldBL_d);

                % oldB_l = obj.FindFixedPointLeft1Fun(oldAL_u, conj(oldAL_d), obj.a, oldA_l);
                % oldB_l = RESHAPE(oldB_l, [obj.chimax, size(obj.b, 1), size(obj.b, 1), obj.chimax]);

                % [oldAL_l, oldBL_l] = obj.BuildLeftCanonicalForm(RESHAPE(oldA_l, [size(oldA_l, 1), size(oldA_l, 2) * size(oldA_l, 3), size(oldA_l, 4)]), ...
                %                                                 RESHAPE(oldB_l, [size(oldB_l, 1), size(oldB_l, 2) * size(oldB_l, 3), size(oldB_l, 4)]), 1e-7, 100);

                % oldAL_l = RESHAPE(oldAL_l, [size(oldA_l, 1), size(oldA_l, 2), size(oldA_l, 3), size(oldA_l, 4)]);
                % oldBL_l = RESHAPE(oldBL_l, [size(oldB_l, 1), size(oldB_l, 2), size(oldB_l, 3), size(oldB_l, 4)]);

                % [oldAR_l, oldBR_l] = obj.BuildRightCanonicalForm(RESHAPE(oldA_l, [size(oldA_l, 1), size(oldA_l, 2) * size(oldA_l, 3), size(oldA_l, 4)]), ...
                %                                                  RESHAPE(oldB_l, [size(oldB_l, 1), size(oldB_l, 2) * size(oldB_l, 3), size(oldB_l, 4)]), 1e-7, 100);

                % oldAR_l = RESHAPE(oldAR_l, [size(oldA_l, 1), size(oldA_l, 2), size(oldA_l, 3), size(oldA_l, 4)]);
                % oldBR_l = RESHAPE(oldBR_l, [size(oldB_l, 1), size(oldB_l, 2), size(oldB_l, 3), size(oldB_l, 4)]);



                % oldA_r = obj.FindFixedPointRight1Fun(oldBL_u, conj(oldBL_d), obj.b, oldB_r);
                % oldA_r = RESHAPE(oldA_r, [obj.chimax, size(obj.a, 3), size(obj.a, 3), obj.chimax]);

                % [oldAL_r, oldBL_r] = obj.BuildLeftCanonicalForm(RESHAPE(oldA_r, [size(oldA_r, 1), size(oldA_r, 2) * size(oldA_r, 3), size(oldA_r, 4)]), ...
                %                                                 RESHAPE(oldB_r, [size(oldB_r, 1), size(oldB_r, 2) * size(oldB_r, 3), size(oldB_r, 4)]), 1e-7, 100);

                % oldAL_r = RESHAPE(oldAL_r, [size(oldA_r, 1), size(oldA_r, 2), size(oldA_r, 3), size(oldA_r, 4)]);
                % oldBL_r = RESHAPE(oldBL_r, [size(oldB_r, 1), size(oldB_r, 2), size(oldB_r, 3), size(oldB_r, 4)]);

                % [oldAR_r, oldBR_r] = obj.BuildRightCanonicalForm(RESHAPE(oldA_r, [size(oldA_r, 1), size(oldA_r, 2) * size(oldA_r, 3), size(oldA_r, 4)]), ...
                %                                                  RESHAPE(oldB_r, [size(oldB_r, 1), size(oldB_r, 2) * size(oldB_r, 3), size(oldB_r, 4)]), 1e-7, 100);

                % oldAR_r = RESHAPE(oldAR_r, [size(oldA_r, 1), size(oldA_r, 2), size(oldA_r, 3), size(oldA_r, 4)]);
                % oldBR_r = RESHAPE(oldBR_r, [size(oldB_r, 1), size(oldB_r, 2), size(oldB_r, 3), size(oldB_r, 4)]);

                [oldAL_l, ~, oldBL_l, ~] = obj.VUMPS(oldAL_l, oldAR_l, oldBL_l, oldBR_l, permute(obj.a, [4, 1, 2, 3, 5]), permute(obj.b, [4, 1, 2, 3, 5]), "both", vumps_tol, LanczosD);
                [oldAL_r, ~, oldBL_r, ~] = obj.VUMPS(oldAL_r, oldAR_r, oldBL_r, oldBR_r, permute(obj.a, [2, 3, 4, 1, 5]), permute(obj.b, [2, 3, 4, 1, 5]), "both", vumps_tol, LanczosD);

                
                
                % fprintf("Calculating density matrix for layer #%d\n", layer);
                layer = layer + 1;
                [dm_lr, oldA_u, oldB_d] = obj.DensityMatrixVUMPS(oldAL_r, oldBL_r, oldAL_l, oldBL_l, permute(obj.a, [2, 3, 4, 1, 5]), permute(obj.b, [2, 3, 4, 1, 5]), LanczosD);
                

                energy_ud = real(trace(H * dm_ud));
                energy_lr = real(trace(H * dm_lr));

                energy_diff_ud = energy_old_ud - energy_ud;
                energy_diff_lr = energy_old_lr - energy_lr;
                diff_dm_ud = dm_ud - olddm_ud;
                diff_dm_lr = dm_lr - olddm_lr;

                % fprintf("density matrix error (ud): %.9e\n", norm(diff_dm_ud));
                % fprintf("Energy error (ud): %.9e\n", norm(energy_diff_ud));
                % fprintf("Current Energy (ud): %.9e\n", trace(H * dm_ud));
                % fprintf("density matrix error (lr): %.9e\n", norm(diff_dm_lr));
                % fprintf("Energy error (lr): %.9e\n", norm(energy_diff_lr));
                % fprintf("Current Energy (lr): %.9e\n", trace(H * dm_lr));

                dm = (dm_ud + dm_lr) / 2;
                % dm = dm_ud;
                % energy_diff_lr = energy_diff_ud;
                % 

                if max(abs(energy_diff_lr), abs(energy_diff_ud)) < vumps_e_tol
                    break;
                else
                    olddm_ud = dm_ud;
                    energy_old_ud = energy_ud;
                    olddm_lr = dm_lr;
                    energy_old_lr = energy_lr;
                end
            

                if layer > 30
                    % fprintf("Reach maximum layer!\n")
                    break
                end

                % Prepare for the next iteration
                

                % oldB_u = obj.FindFixedPointLeft1Fun(oldAL_r, conj(oldAL_l), permute(obj.a, [2, 3, 4, 1, 5]), oldA_u);
                % oldB_u = RESHAPE(oldB_u, [obj.chimax, size(obj.b, 2), size(obj.b, 2), obj.chimax]);

                % [oldAL_u, oldBL_u] = obj.BuildLeftCanonicalForm(RESHAPE(oldA_u, [size(oldA_u, 1), size(oldA_u, 2) * size(oldA_u, 3), size(oldA_u, 4)]), ...
                %                                                 RESHAPE(oldB_u, [size(oldB_u, 1), size(oldB_u, 2) * size(oldB_u, 3), size(oldB_u, 4)]), 1e-7, 100);

                % oldAL_u = RESHAPE(oldAL_u, [size(oldA_u, 1), size(oldA_u, 2), size(oldA_u, 3), size(oldA_u, 4)]);
                % oldBL_u = RESHAPE(oldBL_u, [size(oldB_u, 1), size(oldB_u, 2), size(oldB_u, 3), size(oldB_u, 4)]);

                % [oldAR_u, oldBR_u] = obj.BuildRightCanonicalForm(RESHAPE(oldA_u, [size(oldA_u, 1), size(oldA_u, 2) * size(oldA_u, 3), size(oldA_u, 4)]), ...
                %                                                  RESHAPE(oldB_u, [size(oldB_u, 1), size(oldB_u, 2) * size(oldB_u, 3), size(oldB_u, 4)]), 1e-7, 100);

                % oldAR_u = RESHAPE(oldAR_u, [size(oldA_u, 1), size(oldA_u, 2), size(oldA_u, 3), size(oldA_u, 4)]);
                % oldBR_u = RESHAPE(oldBR_u, [size(oldB_u, 1), size(oldB_u, 2), size(oldB_u, 3), size(oldB_u, 4)]);



                % oldA_d = obj.FindFixedPointRight1Fun(oldBL_r, conj(oldBL_l), permute(obj.b, [2, 3, 4, 1, 5]), oldB_d);
                % oldA_d = RESHAPE(oldA_d, [obj.chimax, size(obj.a, 2), size(obj.a, 2), obj.chimax]);

                % [oldAL_d, oldBL_d] = obj.BuildLeftCanonicalForm(RESHAPE(oldA_d, [size(oldA_d, 1), size(oldA_d, 2) * size(oldA_d, 3), size(oldA_d, 4)]), ...
                %                                                 RESHAPE(oldB_d, [size(oldB_d, 1), size(oldB_d, 2) * size(oldB_d, 3), size(oldB_d, 4)]), 1e-7, 100);

                % oldAL_d = RESHAPE(oldAL_d, [size(oldA_d, 1), size(oldA_d, 2), size(oldA_d, 3), size(oldA_d, 4)]);
                % oldBL_d = RESHAPE(oldBL_d, [size(oldB_d, 1), size(oldB_d, 2), size(oldB_d, 3), size(oldB_d, 4)]);

                % [oldAR_d, oldBR_d] = obj.BuildRightCanonicalForm(RESHAPE(oldA_d, [size(oldA_d, 1), size(oldA_d, 2) * size(oldA_d, 3), size(oldA_d, 4)]), ...
                %                                                  RESHAPE(oldB_d, [size(oldB_d, 1), size(oldB_d, 2) * size(oldB_d, 3), size(oldB_d, 4)]), 1e-7, 100);

                % oldAR_d = RESHAPE(oldAR_d, [size(oldA_d, 1), size(oldA_d, 2), size(oldA_d, 3), size(oldA_d, 4)]);
                % oldBR_d = RESHAPE(oldBR_d, [size(oldB_d, 1), size(oldB_d, 2), size(oldB_d, 3), size(oldB_d, 4)]);
                
            end

            % Sz = [[1, 0]; [0, -1]];
            % Sx = [[0, 1]; [1, 0]];
            % id = [[1, 0]; [0, 1]];

            % Cor_1_dm = obj.CorrelationFunctionDM(obj.a, obj.b, 1, oldA_l, oldB_r, oldAL_u, oldBL_u, oldAL_d, oldBL_d);
            % Cor_2_dm = obj.CorrelationFunctionDM(obj.a, obj.b, 2, oldA_l, oldB_r, oldAL_u, oldBL_u, oldAL_d, oldBL_d);
            % Cor_3_dm = obj.CorrelationFunctionDM(obj.a, obj.b, 3, oldA_l, oldB_r, oldAL_u, oldBL_u, oldAL_d, oldBL_d);
            % Cor_4_dm = obj.CorrelationFunctionDM(obj.a, obj.b, 4, oldA_l, oldB_r, oldAL_u, oldBL_u, oldAL_d, oldBL_d);
            % Cor_5_dm = obj.CorrelationFunctionDM(obj.a, obj.b, 5, oldA_l, oldB_r, oldAL_u, oldBL_u, oldAL_d, oldBL_d);
            % Cor_6_dm = obj.CorrelationFunctionDM(obj.a, obj.b, 6, oldA_l, oldB_r, oldAL_u, oldBL_u, oldAL_d, oldBL_d);
            % Cor_7_dm = obj.CorrelationFunctionDM(obj.a, obj.b, 7, oldA_l, oldB_r, oldAL_u, oldBL_u, oldAL_d, oldBL_d);
            % Cor_8_dm = obj.CorrelationFunctionDM(obj.a, obj.b, 8, oldA_l, oldB_r, oldAL_u, oldBL_u, oldAL_d, oldBL_d);
            % Cor_9_dm = obj.CorrelationFunctionDM(obj.a, obj.b, 9, oldA_l, oldB_r, oldAL_u, oldBL_u, oldAL_d, oldBL_d);
            % Cor_10_dm = obj.CorrelationFunctionDM(obj.a, obj.b, 10, oldA_l, oldB_r, oldAL_u, oldBL_u, oldAL_d, oldBL_d);

            % CZZ_con = [obj.CorrelationFunction(Sz, Sz, Cor_1_dm), ...
            %            obj.CorrelationFunction(Sz, Sz, Cor_2_dm), ...
            %            obj.CorrelationFunction(Sz, Sz, Cor_3_dm), ...
            %            obj.CorrelationFunction(Sz, Sz, Cor_4_dm), ...
            %            obj.CorrelationFunction(Sz, Sz, Cor_5_dm), ...
            %            obj.CorrelationFunction(Sz, Sz, Cor_6_dm), ...
            %            obj.CorrelationFunction(Sz, Sz, Cor_7_dm), ...
            %            obj.CorrelationFunction(Sz, Sz, Cor_8_dm), ...
            %            obj.CorrelationFunction(Sz, Sz, Cor_9_dm), ...
            %            obj.CorrelationFunction(Sz, Sz, Cor_10_dm)];

            % CXX_con = [obj.CorrelationFunction(Sx, Sx, Cor_1_dm), ...
            %            obj.CorrelationFunction(Sx, Sx, Cor_2_dm), ...
            %            obj.CorrelationFunction(Sx, Sx, Cor_3_dm), ...
            %            obj.CorrelationFunction(Sx, Sx, Cor_4_dm), ...
            %            obj.CorrelationFunction(Sx, Sx, Cor_5_dm), ...
            %            obj.CorrelationFunction(Sx, Sx, Cor_6_dm), ...
            %            obj.CorrelationFunction(Sx, Sx, Cor_7_dm), ...
            %            obj.CorrelationFunction(Sx, Sx, Cor_8_dm), ...
            %            obj.CorrelationFunction(Sx, Sx, Cor_9_dm), ...
            %            obj.CorrelationFunction(Sx, Sx, Cor_10_dm)];

            % CZZ = [trace(Cor_1_dm * kron(Sz, Sz)), ...
            %        trace(Cor_2_dm * kron(Sz, Sz)), ...
            %        trace(Cor_3_dm * kron(Sz, Sz)), ...
            %        trace(Cor_4_dm * kron(Sz, Sz)), ...
            %        trace(Cor_5_dm * kron(Sz, Sz)), ...
            %        trace(Cor_6_dm * kron(Sz, Sz)), ...
            %        trace(Cor_7_dm * kron(Sz, Sz)), ...
            %        trace(Cor_8_dm * kron(Sz, Sz)), ...
            %        trace(Cor_9_dm * kron(Sz, Sz)), ...
            %        trace(Cor_10_dm * kron(Sz, Sz))];

            % CXX = [trace(Cor_1_dm * kron(Sx, Sx)), ...
            %        trace(Cor_2_dm * kron(Sx, Sx)), ...
            %        trace(Cor_3_dm * kron(Sx, Sx)), ...
            %        trace(Cor_4_dm * kron(Sx, Sx)), ...
            %        trace(Cor_5_dm * kron(Sx, Sx)), ...
            %        trace(Cor_6_dm * kron(Sx, Sx)), ...
            %        trace(Cor_7_dm * kron(Sx, Sx)), ...  
            %        trace(Cor_8_dm * kron(Sx, Sx)), ...
            %        trace(Cor_9_dm * kron(Sx, Sx)), ...
            %        trace(Cor_10_dm * kron(Sx, Sx))];
            
            % % fprintf("%.9e, ", real(CZZ_con))
            % % fprintf("\n")
            % % fprintf("%.9e, ", real(CXX_con))
            % % fprintf("\n")
            % % fprintf("%.9e, ", real(CZZ))
            % % fprintf("\n")
            % % fprintf("%.9e, ", real(CXX))
            % % fprintf("\n")

            % % disp(obj.CorrelationFunction(Sz, Sz, Cor_1_dm))
            % % disp(obj.CorrelationFunction(Sx, Sx, Cor_1_dm))
            % % disp(obj.CorrelationFunction(Sz, Sz, Cor_2_dm))
            % % disp(obj.CorrelationFunction(Sx, Sx, Cor_2_dm)) 
            % % disp(obj.CorrelationFunction(Sz, Sz, Cor_3_dm)) 
            % % disp(obj.CorrelationFunction(Sx, Sx, Cor_3_dm))
            % % disp(obj.CorrelationFunction(Sz, Sz, Cor_4_dm)) 
            % % disp(obj.CorrelationFunction(Sx, Sx, Cor_4_dm))
            % % disp(obj.CorrelationFunction(Sz, Sz, Cor_5_dm)) 
            % % disp(obj.CorrelationFunction(Sx, Sx, Cor_5_dm))
            % % disp(obj.CorrelationFunction(Sz, Sz, Cor_6_dm)) 
            % % disp(obj.CorrelationFunction(Sx, Sx, Cor_6_dm))
            % % disp(obj.CorrelationFunction(Sz, Sz, Cor_7_dm))
            % % disp(obj.CorrelationFunction(Sx, Sx, Cor_7_dm))  

            % % disp(trace(Cor_1_dm * kron(Sz, Sz)))
            % % disp(trace(Cor_1_dm * kron(Sx, Sx)))
            % % disp(trace(Cor_2_dm * kron(Sz, Sz)))
            % % disp(trace(Cor_2_dm * kron(Sx, Sx)))
            % % disp(trace(Cor_3_dm * kron(Sz, Sz)))
            % % disp(trace(Cor_3_dm * kron(Sx, Sx)))
            % % disp(trace(Cor_4_dm * kron(Sz, Sz)))
            % % disp(trace(Cor_4_dm * kron(Sx, Sx)))
            % % disp(trace(Cor_5_dm * kron(Sz, Sz)))
            % % disp(trace(Cor_5_dm * kron(Sx, Sx)))
            % % disp(trace(Cor_6_dm * kron(Sz, Sz)))
            % % disp(trace(Cor_6_dm * kron(Sx, Sx)))
            % % disp(trace(Cor_7_dm * kron(Sz, Sz)))
            % % disp(trace(Cor_7_dm * kron(Sx, Sx)))
        end

        function cor = CorrelationFunction(obj, op1, op2, dm)
            id = eye(obj.d);
            cor = trace(dm * kron(op1, op2)) - trace(dm * kron(op1, id)) * trace(dm * kron(id, op2));
        end

        function dm = VUMPS1test(obj, AL, AR)
            oldAL_u = AL;
            oldAR_u = AR;
            
            oldAL_d = AL;
            oldAR_d = AR;

            olddm = zeros(4, 4);
            
            while true

                [newAL_u, newAR_u] = obj.VUMPS1(oldAL_u, oldAR_u, obj.a);
                [newAL_d, newAR_d] = obj.VUMPS1(oldAL_d, oldAR_d, permute(obj.a, [3, 4, 1, 2, 5]));

                oldAL_u = newAL_u;
                oldAR_u = newAR_u;
                
                oldAL_d = newAL_d;
                oldAR_d = newAR_d;

                dm = obj.DensityMatrixVUMPS(oldAL_u, oldAL_u, oldAL_d, oldAL_d, obj.a, obj.a);

                diff_dm = olddm - dm;
                if norm(diff_dm) < 1e-6
                    break;
                else
                    % fprintf("density matrix error: %.9e\n", norm(diff_dm));
                    olddm = dm;
                end

            end
        end

        function CTMR(obj, ctm_tol, svd_tol, pinv_tol, H, normalize, initialize)
            
            if initialize == 1
                obj.CTMInitialize(obj, "eye")
                
            end

            energy_error = inf;
            energy_old = abs(obj.Energy(obj, H, 1));

            iter = 0;
            
            while (energy_error > ctm_tol) && iter < 100
                

                for jj = 1:5
                    for ii = [3, 1, 0, 2]
                        % fprintf("Bond Direction: %d\n", ii)

                        obj.CTMRenorm(obj, ii, svd_tol, pinv_tol, normalize, 1);
                    end
                end
                
                
                energy_new = abs(obj.Energy(obj, H, 1));
                energy_error = abs(energy_new - energy_old);
                % fprintf("Old Energy: %.9e\n", energy_old);
                % fprintf("Energy error: %.9e\n", energy_error);
                % fprintf("New Energy: %.9e\n", energy_new);
                energy_old = energy_new;
                iter = iter + 5;
                % fprint("Num of iter: %d\n", iter);
            end
            % fprintf("total iteration: %d\n", iter);
        end

        function CTMR_QR(obj, obj_low, ctm_tol, ctm_svd_tol, H, normalize, initialize, InitVUMPS, P)
            
            if initialize == 1
                obj.CTMInitialize(obj_low, "eye") 
            end

            energy_error = inf;
            energy_old = real(obj.Energy(obj_low, H, 1));

            iter = 0;

            while (energy_error > ctm_tol) && iter < 100
                
                for jj = 1:2
                    for ii = [0, 1, 2, 3]
                        % fprintf("Bond dimension: %d\n", ii);
                        corners = obj.CTMRenormCornersQR(obj_low, ii);
                        obj.CTMRenorm_QR_(obj_low, corners, ii, ctm_svd_tol, normalize, 1, InitVUMPS);
                    end
                end

                % for jj = 1:2
                %     for ii = [1, 3]
                %         % fprintf("Bond dimension: %d\n", ii);
                %         corners = obj.CTMRenormCornersQR(obj_low, ii);
                %         obj.CTMRenorm_QR_(obj_low, corners, ii, ctm_svd_tol, normalize, 1, InitVUMPS);
                %     end
                % end

                
                % for jj = 1:3
                %     for ii = [0, 2, 1, 3]
                %         % fprintf("Bond direction: %d\n", ii);
                %         % fprintf("Build Corners\n")
                %         corners = obj.CTMRenormCornersQR(obj_low, ii);
                %         % corners = obj.CTMRenormSingleLayerCornersQR(obj_low, ii);
                %         obj.CTMRenormSingleLayerQR(obj_low, corners, ii, ctm_svd_tol, normalize, 1);
                %     end
                % end
                
                % fprintf("Calculating Energy\n");
                [energy_new, rho] = obj.Energy(obj_low, H, 1);
                % fprintf("%e ", eig(rho));
                % fprintf("\n")
                energy_new = real(energy_new);
                energy_error = abs(energy_new - energy_old);
                % fprintf("Old Energy: %.9e\n", energy_old);
                % fprintf("Energy error: %.9e\n", energy_error);
                % fprintf("New Energy: %.9e\n", energy_new);
                energy_old = energy_new;
                iter = iter + 2;
            end
            % fprintf("total iteration: %d\n", iter);
        end


        function [energy, rhoAB] = Energy(obj, obj_low, H, normalize)

            energy_new = 0;
            rhoAB = 0;

            for ii = 0:3
                [temp_energy, temp_rho] = obj.ExpectationValue(obj_low, H, ii, normalize);
                % fprintf("bond_dir: %d, energy: %.9e\n", [ii, temp_energy])
                energy_new = energy_new + real(temp_energy);
                rhoAB = rhoAB + temp_rho;
            end
            energy_new = energy_new / 4.0;
            rhoAB = rhoAB / 4.0;
            % % disp(['deltaE', deltaE])
            energy = energy_new;
            
        end

        function [energy, rho] = BangBang_Ising(obj, H1, it_rt, ntu_tol, svd_tol, pinv_tol, ctm_tol, P, change_peps, x)
            
            alpha = x(1: P + 1);
            beta = x(P + 2: 2 * P + 1);

            % H_ising = (kron(H1, eye(2)) + kron(eye(2), H1)) / 4.0 - kron([[1, 0];[0, -1]], [[1, 0];[0, -1]]);
            
            if change_peps == 0
                obj_temp = copy(obj);
            else
                obj_temp = obj;
            end

            for ii = 1:P + 1

                if it_rt == 0
                    Ulocal = transpose(expm(-H1 * alpha(ii)));
                else
                    Ulocal = transpose(expm(-H1 * alpha(ii) * 1j));
                end
                obj_temp.ApplyLocalGate(Ulocal, 0);
                obj_temp.ApplyLocalGate(Ulocal, 1);
                

                if ii < P + 1
                    [U_left, U_right] = obj_temp.HamiltonianExp_Ising(beta(ii), it_rt);
                    for bond_dir = [0, 2, 1, 3]
                        obj_temp.ApplyGate(U_left, U_right, bond_dir, "NTU", ntu_tol, svd_tol, pinv_tol);
                    end
                end
            end
            
            obj_temp.CTMR(obj_temp, ctm_tol, svd_tol, pinv_tol, (kron(H1, eye(obj.d)) + kron(eye(obj.d), H1)) / 4 - ...
                kron([[1, 0]; [0, -1]], [[1, 0]; [0, -1]]), 1, 1)
            [energy, rho] = obj_temp.Energy(obj_temp, (kron(H1, eye(obj.d)) + kron(eye(obj.d), H1)) / 4 - ...
                kron([[1, 0]; [0, -1]], [[1, 0]; [0, -1]]), 1);
            if change_peps == 0
                delete(obj_temp)
            end
        end

        function Periodic2x2test(obj)

            Sz = [[1, 0]; [0, -1]];
            Sx = [[0, 1]; [1, 0]];
            Id = [[1, 0]; [0, 1]];
            temp = tensorprod(obj.a, obj.b, [1, 3], [3, 1], NumDimensionsA=5);
            temp = tensorprod(temp, obj.b, [2, 1], [2, 4], NumDimensionsA=6);
            temp = tensorprod(temp, obj.a, [6, 3, 5, 2], [1, 2, 3, 4], NumDimensionsA=7);

            norm = tensorprod(temp, conj(temp), [1, 2, 3, 4], [1, 2, 3, 4], NumDimensionsA=4);

            dir_0_IX = tensorprod(temp, conj(temp), [3, 4], [3, 4], NumDimensionsA=4);
            dir_0_IX = tensorprod(tensorprod(dir_0_IX, Id, [1, 3], [1, 2]), Sx, [1, 2], [1, 2]) / norm;

            dir_1_IX = tensorprod(temp, conj(temp), [2, 4], [2, 4], NumDimensionsA=4);
            dir_1_IX = tensorprod(tensorprod(dir_1_IX, Id, [1, 3], [1, 2]), Sx, [1, 2], [1, 2]) / norm;

            dir_2_IX = tensorprod(temp, conj(temp), [1, 2], [1, 2], NumDimensionsA=4);
            dir_2_IX = tensorprod(tensorprod(dir_2_IX, Id, [1, 3], [1, 2]), Sx, [1, 2], [1, 2]) / norm;

            dir_3_IX = tensorprod(temp, conj(temp), [1, 3], [1, 3], NumDimensionsA=4);
            dir_3_IX = tensorprod(tensorprod(dir_3_IX, Id, [1, 3], [1, 2]), Sx, [1, 2], [1, 2]) / norm;

            dir_0_ZZ = tensorprod(temp, conj(temp), [3, 4], [3, 4], NumDimensionsA=4);
            dir_0_ZZ = tensorprod(tensorprod(dir_0_ZZ, Sz, [1, 3], [1, 2]), Sz, [1, 2], [1, 2]) / norm;

            dir_1_ZZ = tensorprod(temp, conj(temp), [2, 4], [2, 4], NumDimensionsA=4);
            dir_1_ZZ = tensorprod(tensorprod(dir_1_ZZ, Sz, [1, 3], [1, 2]), Sz, [1, 2], [1, 2]) / norm;

            dir_2_ZZ = tensorprod(temp, conj(temp), [1, 2], [1, 2], NumDimensionsA=4);
            dir_2_ZZ = tensorprod(tensorprod(dir_2_ZZ, Sz, [1, 3], [1, 2]), Sz, [1, 2], [1, 2]) / norm;

            dir_3_ZZ = tensorprod(temp, conj(temp), [1, 3], [1, 3], NumDimensionsA=4);
            dir_3_ZZ = tensorprod(tensorprod(dir_3_ZZ, Sz, [1, 3], [1, 2]), Sz, [1, 2], [1, 2]) / norm;

            % % disp(dir_0_ZZ)
            % % disp(dir_1_ZZ)
            % % disp(dir_2_ZZ)
            % % disp(dir_3_ZZ)

            % % disp(dir_0_IX)
            % % disp(dir_1_IX)
            % % disp(dir_2_IX)
            % % disp(dir_3_IX)

            % % disp(norm)
        end

        

        function [energy, rho] = BangBang_Ising_(obj, H1, H, it_rt, ntu_tol, svd_tol, pinv_tol, ctm_tol, ctm_svd_tol, P, change_peps, x, ctm, rand_gauge)
            
            alpha_ = x(1: P);
            beta_ = x(P + 1: 2 * P);
            % H1_ = -[[0, 1]; [1, 0]];
            % fprintf("%.9e, ", alpha_);
            % fprintf("\n");
            % fprintf("%.9e, ", beta_);
            % fprintf("\n");

            % H_ising = (kron(H1, eye(2)) + kron(eye(2), H1)) / 4.0 - kron([[1, 0];[0, -1]], [[1, 0];[0, -1]]);
            
            if change_peps == 0
                obj_temp = copy(obj);
            else
                obj_temp = obj;
            end

            total_ntu_error = 0;

            [U_left_0, U_right_0] = obj_temp.HamiltonianExp_Ising(0, it_rt);

            for ii = 1:P

                % fprintf("Bang-bang # %d\n", ii)

                [U_left, U_right] = obj_temp.HamiltonianExp_Ising(beta_(ii), it_rt);
                
                for bond_dir = [0, 2, 1, 3]
                    obj_temp.ApplyGate(U_left, U_right, bond_dir, "Notrunc", ntu_tol, svd_tol, pinv_tol);
                end
                
                for bond_dir = [0, 2, 1, 3]
                    ntu_error = obj_temp.ApplyGate(U_left_0, U_right_0, bond_dir, "NTU", ntu_tol, svd_tol, pinv_tol);
                    % fprintf("NTU error at this step:%.9e\n", ntu_error)
                    total_ntu_error = total_ntu_error + ntu_error;
                end

                obj_temp.ApplyLocalGate([[0, 1]; [1, 0]], 0);
                obj_temp.ApplyLocalGate([[0, 1]; [1, 0]], 1);

                if it_rt == 0
                    Ulocal = transpose(expm(-H1 * alpha_(ii)));
                else
                    Ulocal = transpose(expm(-H1 * alpha_(ii) * 1j));
                end
                obj_temp.ApplyLocalGate(Ulocal, 0);
                obj_temp.ApplyLocalGate(Ulocal, 1);
                % for bond_dir = [0, 1, 2, 3]
                %     obj_temp.ApplyGate(U_left_0, U_right_0, bond_dir, "NTU", ntu_tol, svd_tol, pinv_tol);
                % end
            end

            % fprintf("NTU done\n")
            % fprintf("Accumulated NTU error: %.9e\n", total_ntu_error)
            sum_a = tensorprod(obj_temp.a, conj(obj_temp.a), [1,2,3,4,5], [1,2,3,4,5]);
            sum_b = tensorprod(obj_temp.b, conj(obj_temp.b), [1,2,3,4,5], [1,2,3,4,5]);
            % fprintf("sum_a: %.9e\n", sum_a);
            % fprintf("sum_b: %.9e\n", sum_b);

            obj_temp.Cubize();

            if rand_gauge == 1
                obj_temp.RandomGauge(100, 7);
            end

            if ctm == 1
                % fprintf("CTMRG begin\n")
                % obj_temp.CTMR_QR(obj_temp, ctm_tol, ctm_svd_tol, ctm_tol, (kron(H1, eye(obj.d)) + kron(eye(obj.d), H1)) / 4 - ...
                %     kron([[1, 0]; [0, -1]], [[1, 0]; [0, -1]]), 1, 1)
                % obj_temp.CTMR_QR(obj_temp, ctm_tol, ctm_svd_tol, (kron(H1, eye(obj.d)) + kron(eye(obj.d), H1)) / 4 - ...
                %     kron([[1, 0]; [0, -1]], [[1, 0]; [0, -1]]), 1, 1, 0, P);
                obj_temp.CTMR(ctm_tol, ctm_svd_tol, pinv_tol, (kron(H1, eye(obj.d)) + kron(eye(obj.d), H1)) / 4 - ...
                    kron([[1, 0]; [0, -1]], [[1, 0]; [0, -1]]), 1, 1);
                % fprintf("CTM done\n")
                % obj_temp.EnlargeCTM();
                [energy, rho] = obj_temp.Energy(obj_temp, H, 1);
                % disp(rho)
                % fprintf("Energy: %.9e\n", energy);
            elseif ctm == 2

                % obj_temp.CTMInitialize(obj_temp, "eye", 1)

                vumps_e_tol = ctm_tol;
                vumps_tol = ctm_svd_tol;

                % sum_a = tensorprod(obj_temp.a, conj(obj_temp.a), [1,2,3,4,5], [1,2,3,4,5]);
                % sum_b = tensorprod(obj_temp.b, conj(obj_temp.b), [1,2,3,4,5], [1,2,3,4,5]);
                % obj_temp.a = obj_temp.a / sqrt(sum_a);
                % obj_temp.b = obj_temp.b / sqrt(sum_b);
                % obj_temp.CTMR_QR(obj_temp, ctm_tol, ctm_svd_tol, (kron(H1, eye(obj.d)) + kron(eye(obj.d), H1)) / 4 - ...
                %     kron([[1, 0]; [0, -1]], [[1, 0]; [0, -1]]), 1, 1, 1);
                    
                [rho, TA1, TB1, TA3, TB3] = obj_temp.VUMPStest(vumps_tol, vumps_e_tol, H, 2 * P);
                obj_temp.TA{1} = TA1;
                obj_temp.TA{3} = TA3;
                obj_temp.TB{1} = TB1;
                obj_temp.TB{3} = TB3;
                % % disp(rho)
                energy = trace(H * rho);
                energy = real(energy)
                fprintf("%.9e\n", energy);

            end
            
            
            if change_peps == 0
                delete(obj_temp)
            end
        end

        function RandomGauge(obj, Singularity, k)

            % For test, even bond dimension only

            S1 = [rand(1, k) / Singularity, rand(1, size(obj.a, 1) - k) * Singularity]; 
            S2 = [rand(1, k) / Singularity, rand(1, size(obj.a, 2) - k) * Singularity]; 
            S3 = [rand(1, k) / Singularity, rand(1, size(obj.a, 3) - k) * Singularity]; 
            S4 = [rand(1, k) / Singularity, rand(1, size(obj.a, 4) - k) * Singularity];

            [U1, ~] = qr(rand(size(obj.a, 1), size(obj.a, 1)));
            [U2, ~] = qr(rand(size(obj.a, 2), size(obj.a, 2)));
            [U3, ~] = qr(rand(size(obj.a, 3), size(obj.a, 3)));
            [U4, ~] = qr(rand(size(obj.a, 4), size(obj.a, 4)));

            [V1, ~] = qr(rand(size(obj.a, 1), size(obj.a, 1)));
            [V2, ~] = qr(rand(size(obj.a, 2), size(obj.a, 2)));
            [V3, ~] = qr(rand(size(obj.a, 3), size(obj.a, 3)));
            [V4, ~] = qr(rand(size(obj.a, 4), size(obj.a, 4)));

            Q1 = U1 * diag(S1) * (V1');
            Q2 = U2 * diag(1.0 ./ S2) * (V2');
            Q3 = U3 * diag(S3) * (V3');
            Q4 = U4 * diag(1.0 ./ S4) * (V4');

            Q1_ = V1 * diag(1.0 ./ S1) * (U1');
            Q2_ = V2 * diag(S2) * (U2');
            Q3_ = V3 * diag(1.0 ./ S3) * (U3');
            Q4_ = V4 * diag(S4) * (U4');

            obj.a = tensorprod(obj.a, Q1, 1, 1);
            obj.a = tensorprod(obj.a, Q2, 1, 1);
            obj.a = tensorprod(obj.a, Q3, 1, 1);
            obj.a = tensorprod(obj.a, Q4, 1, 1);
            obj.a = permute(obj.a, [2, 3, 4, 5, 1]);
            
            obj.b = tensorprod(Q3_, obj.b, 2, 1);
            obj.b = tensorprod(Q4_, obj.b, 2, 2);
            obj.b = tensorprod(Q1_, obj.b, 2, 3);
            obj.b = tensorprod(Q2_, obj.b, 2, 4);

            obj.b = permute(obj.b, [4, 3, 2, 1, 5]);
        end

        function [energy, rho] = BangBang_Ising_inverse_(obj, H1, it_rt, ntu_tol, svd_tol, pinv_tol, ctm_tol, P, change_peps, x, do_ctm)
            
            alpha = x(1: P);
            beta = x(P + 1: 2 * P);
            % H1_ = -[[0, 1]; [1, 0]];
            % fprintf("%.9e, ", alpha);
            % fprintf("\n");
            % fprintf("%.9e, ", beta);
            % fprintf("\n");

            % H_ising = (kron(H1, eye(2)) + kron(eye(2), H1)) / 4.0 - kron([[1, 0];[0, -1]], [[1, 0];[0, -1]]);
            
            if change_peps == 0
                obj_temp = copy(obj);
            else
                obj_temp = obj;
            end

            for ii = P:-1:1

                if it_rt == 0
                    Ulocal = transpose(expm(-H1 * alpha(ii)));
                else
                    Ulocal = transpose(expm(H1 * alpha(ii) * 1j));
                end
                obj_temp.ApplyLocalGate(Ulocal, 0);
                obj_temp.ApplyLocalGate(Ulocal, 1);

                [U_left, U_right] = obj_temp.HamiltonianExp_Ising(-beta(ii), it_rt);
                for bond_dir = 0:3
                    obj_temp.ApplyGate(U_left, U_right, bond_dir, "NTU", ntu_tol, svd_tol, pinv_tol);
                end
            end

            [U_left, U_right] = obj_temp.HamiltonianExp_Ising(0, 1);
            for bond_dir = 0:3
                obj_temp.ApplyGate(U_left, U_right, bond_dir, "NTU", ntu_tol, svd_tol, pinv_tol)
            end

            
            % fprintf("NTU done\n")
            if do_ctm == 1
                % fprintf("CTMRG begin\n")
                % obj_temp.CTMR(obj_temp, ctm_tol, svd_tol, pinv_tol, (kron(H1, eye(obj.d)) + kron(eye(obj.d), H1)) / 4 - ...
                %     kron([[1, 0]; [0, -1]], [[1, 0]; [0, -1]]), 1, 1);
                obj_temp.CTMR_QR(obj_temp, ctm_tol, ctm_svd_tol, (kron(H1, eye(obj.d)) + kron(eye(obj.d), H1)) / 4 - ...
                                kron([[1, 0]; [0, -1]], [[1, 0]; [0, -1]]), 1, 1);
                % fprintf("CTM done\n")
                % fprintf("Calculating Energy\n")
                [energy, rho] = obj_temp.Energy(obj_temp, (kron(H1, eye(obj.d)) + kron(eye(obj.d), H1)) / 4 - ...
                    kron([[1, 0]; [0, -1]], [[1, 0]; [0, -1]]), 1);
            else
                energy = 0;
                rho = 0;
            end
            
            
            if change_peps == 0
                delete(obj_temp)
            end
        end
        
        function [energy, rho, obj_temp] = GroundStateSearch(obj, H1, tau0, Etol, ctm_tol, ctm_svd_tol, svd_tol, pinv_tol, ntu_tol, ctm)
            obj_temp = copy(obj);

            deltaE = inf;
            tau = tau0;
            vumps_e_tol = ctm_tol;
            vumps_tol = ctm_svd_tol;

            H = (kron(H1, eye(obj.d)) + kron(eye(obj.d), H1)) / 4 - ...
                kron([[1, 0]; [0, -1]], [[1, 0]; [0, -1]]);

            % rho = obj_temp.VUMPStest(vumps_tol, vumps_e_tol, H, 3);
            % energy = trace(H * rho);
            % energy = real(energy);
            % [energy, rho] = obj_temp.Energy(obj_temp, H, 1);
            rho = ones(4,4);
            energy = inf;

            % fprintf("Searching\n");
            while deltaE > Etol
                % fprintf("Current tau: %.9e\n", tau);
                temp = copy(obj_temp);
                [U_left, U_right] = temp.HamiltonianExp(H, tau, 0);
                % [U_left, U_right] = temp.HamiltonianExp_Ising(tau, 0);
                for ii = 1:1
                    for jj = [0, 2, 1, 3]
                        temp.ApplyGate(U_left, U_right, jj, "NTU", ntu_tol, svd_tol, pinv_tol);
                    end
                end

                % Ulocal = transpose(expm(-H1 * tau));
                % temp.ApplyLocalGate(Ulocal, 0);
                % temp.ApplyLocalGate(Ulocal, 1);

                % fprintf("NTU done\n");

                temp.Cubize();
    
                if ctm == 1
                    % fprintf("CTMRG begin\n")
                    % obj_temp.CTMR_QR(obj_temp, ctm_tol, ctm_svd_tol, ctm_tol, (kron(H1, eye(obj.d)) + kron(eye(obj.d), H1)) / 4 - ...
                    %     kron([[1, 0]; [0, -1]], [[1, 0]; [0, -1]]), 1, 1)
                    temp.CTMR_QR(temp, ctm_tol, ctm_svd_tol, H, 1, 1, 0);
                    % temp.CTMR(ctm_tol, ctm_svd_tol, pinv_tol, H, 1, 1);
                    % fprintf("CTM done\n")
                    % obj_temp.EnlargeCTM();
                    [energy_new, rho_new] = temp.Energy(temp, H, 1);
                    % fprintf("new Energy: %.9e\n", energy_new);
                elseif ctm == 2
                    
    
                    % sum_a = tensorprod(obj_temp.a, conj(obj_temp.a), [1,2,3,4,5], [1,2,3,4,5]);
                    % sum_b = tensorprod(obj_temp.b, conj(obj_temp.b), [1,2,3,4,5], [1,2,3,4,5]);
                    % obj_temp.a = obj_temp.a / sqrt(sum_a);
                    % obj_temp.b = obj_temp.b / sqrt(sum_b);
                    % obj_temp.CTMR_QR(obj_temp, ctm_tol, ctm_svd_tol, (kron(H1, eye(obj.d)) + kron(eye(obj.d), H1)) / 4 - ...
                    %     kron([[1, 0]; [0, -1]], [[1, 0]; [0, -1]]), 1, 1, 1);
                        
                    rho_new = temp.VUMPStest(vumps_tol, vumps_e_tol, H, 10);
                    energy_new = trace(H * rho_new);
                    energy_new = real(energy_new);
                    % fprintf("new Energy: %.9e\n", energy_new);
    
                end

                deltaE = energy - energy_new;
                % fprintf("deltaE:%.9e\n", deltaE);


                if energy_new <= energy
                    obj_temp = copy(temp);
                    energy = energy_new;
                    rho = rho_new;
                    % fprintf('Current Energy: %.9e\n', energy);
                else
                    tau = tau / 2;
                    deltaE = 100 * Etol;    
                end
                deltaE = abs(deltaE);
            end
        end
    end
end



